"""Validation and the per-tenant data-quality report.

The report is written for a person deciding whether to load: counts, completeness per
field, what was quarantined and why, duplicates, broken references between files, and a
decision (ok, warn, block) with the reasons. It contains row numbers and column names,
never cell values.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date

from .config import DEFAULT_THRESHOLDS
from .normalize import NormalizedFile
from .schema import ENTITIES

PARENT = {("invoice", "customer_id"): "customer", ("payment", "invoice_id"): "invoice"}


@dataclass
class Finding:
    level: str            # info | warning | block
    code: str
    message: str


@dataclass
class DQReport:
    tenant: str
    entities: dict[str, dict] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)

    @property
    def decision(self) -> str:
        levels = {f.level for f in self.findings}
        return "block" if "block" in levels else "warn" if "warning" in levels else "ok"

    def to_dict(self) -> dict:
        return {"tenant": self.tenant, "decision": self.decision, "entities": self.entities,
                "findings": [f.__dict__ for f in self.findings]}

    def render(self) -> str:
        out = [f"Data quality report: {self.tenant}", f"Decision: {self.decision.upper()}", ""]
        for ent, e in self.entities.items():
            out.append(f"{ent} ({', '.join(e['files'])})")
            out.append(f"  rows read {e['rows_in']}, loaded {e['loaded']}, quarantined {e['quarantined']}, "
                       f"duplicate keys {e['duplicates_identical']} identical + {e['duplicates_conflicting']} conflicting (last wins), "
                       f"junk lines skipped {e['dropped_junk']}")
            comp = ", ".join(f"{k} {v:.0%}" for k, v in e["completeness"].items())
            out.append(f"  completeness: {comp}")
            for code, d in sorted(e["issues"].items(), key=lambda kv: -kv[1]["count"]):
                out.append(f"  {code}: {d['count']} (rows {', '.join(map(str, d['examples']))}{', ...' if d['count'] > len(d['examples']) else ''})")
            for why, n in sorted(e["quarantine_reasons"].items(), key=lambda kv: -kv[1])[:4]:
                out.append(f"  quarantined for {why}: {n}")
        out.append("")
        for f in self.findings:
            out.append(f"[{f.level}] {f.code}: {f.message}")
        return "\n".join(out)


def validate(tenant: str, normalized: dict[str, NormalizedFile], existing_keys: dict[str, set] | None = None,
             thresholds: dict | None = None) -> DQReport:
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    existing_keys = existing_keys or {}
    rep = DQReport(tenant)
    by_entity: dict[str, list[NormalizedFile]] = defaultdict(list)
    for nf in normalized.values():
        by_entity[nf.entity].append(nf)
    keys: dict[str, set] = {e: set(existing_keys.get(e, set())) for e in ENTITIES}
    for ent, nfs in by_entity.items():
        recs = [r for nf in nfs for r in nf.records]
        keys[ent] |= {r[f"{ent}_id"] for r in recs}
        issues: dict[str, dict] = {}
        for nf in nfs:
            for i in nf.issues:
                d = issues.setdefault(f"{i.code} in {i.column}", {"count": 0, "examples": []})
                d["count"] += 1
                if len(d["examples"]) < 3:
                    d["examples"].append(i.row)
        reasons: Counter = Counter()
        for nf in nfs:
            for _, why in nf.quarantined:
                reasons[why] += 1
        names = [f.name for f in ENTITIES[ent]]
        comp = {n: (sum(1 for r in recs if r.get(n) is not None) / len(recs) if recs else 0.0) for n in names}
        stats = {k: sum(nf.stats.get(k, 0) for nf in nfs) for k in ("rows_in", "loaded", "quarantined", "duplicates_identical",
                                                                    "duplicates_conflicting", "dropped_junk")}
        stats.update({"files": [nf.filename for nf in nfs], "completeness": comp, "issues": issues, "quarantine_reasons": dict(reasons)})
        rep.entities[ent] = stats
        rate = stats["quarantined"] / stats["rows_in"] if stats["rows_in"] else 0.0
        if rate > th["max_quarantine_rate"]:
            rep.findings.append(Finding("block", "quarantine_rate", f"{ent}: {rate:.0%} of rows quarantined (limit {th['max_quarantine_rate']:.0%}); "
                                                                    "the mapping or a transform is probably wrong"))
        elif stats["quarantined"]:
            rep.findings.append(Finding("warning", "quarantined_rows", f"{ent}: {stats['quarantined']} rows quarantined"))
        for nf in nfs:
            if nf.stats.get("unmapped_required"):
                rep.findings.append(Finding("block", "unmapped_required", f"{nf.filename}: required fields with no source column: "
                                                                         f"{', '.join(nf.stats['unmapped_required'])}"))
        if stats["duplicates_conflicting"]:
            rep.findings.append(Finding("warning", "conflicting_duplicates", f"{ent}: {stats['duplicates_conflicting']} duplicate keys with different content; last row kept"))
        # semantic checks
        bad_dates = neg = 0
        for r in recs:
            if ent == "invoice" and r.get("due_date") and r.get("invoice_date") and date.fromisoformat(r["due_date"]) < date.fromisoformat(r["invoice_date"]):
                bad_dates += 1
            if ent in ("invoice", "payment") and r.get("amount") is not None and float(r["amount"]) <= 0:
                neg += 1
        if bad_dates:
            rep.findings.append(Finding("warning", "due_before_invoice", f"invoice: {bad_dates} rows have a due date before the invoice date"))
        if neg:
            rep.findings.append(Finding("warning", "non_positive_amount", f"{ent}: {neg} rows have an amount of zero or less"))
    # references between files
    for (ent, col), parent in PARENT.items():
        recs = [r for nf in by_entity.get(ent, []) for r in nf.records]
        if not recs or not keys[parent]:
            continue
        orphans = sum(1 for r in recs if r.get(col) is not None and r[col] not in keys[parent])
        rate = orphans / len(recs)
        rep.entities[ent].setdefault("orphans", {})[col] = orphans
        if rate > th["max_orphan_rate"]:
            rep.findings.append(Finding("block", "orphan_references", f"{ent}.{col}: {rate:.0%} of rows point at a {parent} that is not loaded "
                                                                      "(limit {:.0%}); wrong column mapped, or a file is missing".format(th["max_orphan_rate"])))
        elif orphans:
            rep.findings.append(Finding("warning", "orphan_references", f"{ent}.{col}: {orphans} rows point at an unknown {parent}"))
    if not rep.findings:
        rep.findings.append(Finding("info", "clean", "no findings"))
    return rep
