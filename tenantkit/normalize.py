"""Turn a raw table into canonical records using a file's config entry.

Every problem becomes an Issue with a row number, a column and a code, never the raw
value (cells can hold names, emails and phone numbers, and a quality report that
repeats them would be a leak of its own). A row missing a required field, or with a
required field that does not parse, is quarantined instead of loaded. Duplicate keys
inside one file are collapsed, last one wins, and counted as identical or conflicting.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from decimal import Decimal

from .parsing import is_email, is_null, normalize_phone, parse_date, parse_money
from .readers import RawTable
from .schema import COUNTRY_LOOKUP, CURRENCY_CODES, CURRENCY_SYMBOLS, ENTITIES, norm_text


@dataclass
class Issue:
    row: int
    column: str
    code: str
    severity: str = "warning"          # warning | error


@dataclass
class NormalizedFile:
    filename: str
    entity: str
    records: list[dict]
    issues: list[Issue] = field(default_factory=list)
    quarantined: list[tuple[int, str]] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def record_hash(rec: dict) -> str:
    body = {k: v for k, v in rec.items() if not k.startswith("_")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _convert(tr: dict, raw: str, issues: list[Issue], row: int, target: str, required: bool, state: dict):
    """Return the canonical value, or None. Appends to issues. `state` collects side information."""
    if is_null(raw):
        return None
    t = tr.get("type")
    v = raw.strip()
    sev = "error" if required else "warning"
    if t == "id" or t == "text":
        return v
    if t == "email":
        if is_email(v):
            return v.lower()
        issues.append(Issue(row, target, "invalid_email"))
        return None
    if t == "phone":
        p = normalize_phone(v)
        if p is None:
            issues.append(Issue(row, target, "unparseable_phone"))
        return p
    if t == "country":
        iso = COUNTRY_LOOKUP.get(norm_text(v))
        if iso is None:
            issues.append(Issue(row, target, "unknown_country"))
        return iso
    if t == "currency":
        up = v.upper()
        if up in CURRENCY_CODES:
            return up
        if v in CURRENCY_SYMBOLS:
            return CURRENCY_SYMBOLS[v]
        issues.append(Issue(row, target, "unknown_currency", sev))
        return None
    if t == "date":
        d = parse_date(v, tr.get("format", "%Y-%m-%d"))
        if d is None:
            issues.append(Issue(row, target, "unparseable_date", sev))
            return None
        return d.isoformat()
    if t == "money":
        amount, cur = parse_money(v, tr.get("decimal", "."), Decimal(str(tr.get("scale", 1))))
        if amount is None:
            issues.append(Issue(row, target, "unparseable_amount", sev))
            return None
        if cur and tr.get("currency_in_value"):
            state["currency"] = cur
        return f"{amount:.2f}"
    if t in ("status", "method"):
        mapped = {norm_text(k): val for k, val in tr.get("value_map", {}).items()}.get(norm_text(v))
        if mapped is None:
            issues.append(Issue(row, target, f"unknown_{t}"))
        return mapped
    return v


def normalize_table(table: RawTable, fcfg: dict) -> NormalizedFile:
    entity = fcfg["entity"]
    fields = {f.name: f for f in ENTITIES[entity]}
    allow_missing = set(fcfg.get("allow_missing", []))
    idx = {c: i for i, c in enumerate(table.columns)}
    mapped = [(idx[c["source"]], c["target"], c.get("transform", {})) for c in fcfg["columns"] if c.get("target") and c["source"] in idx]
    mapped_targets = {t for _, t, _ in mapped}
    if entity == "invoice" and any(t == "amount" and tr.get("currency_in_value") for _, t, tr in mapped):
        mapped_targets.add("currency")                 # the currency travels inside the amount cells
    key_field = f"{entity}_id"
    out = NormalizedFile(table.name, entity, [])
    seen: dict[str, dict] = {}
    n_ident = n_conf = 0
    for rno, row in enumerate(table.rows, 1):
        rec = {f: None for f in fields}
        issues: list[Issue] = []
        state: dict = {}
        for i, target, tr in mapped:
            rec[target] = _convert(tr, row[i], issues, rno, target, fields[target].required, state)
        if entity == "invoice" and rec.get("currency") is None and state.get("currency"):
            rec["currency"] = state["currency"]
        reasons = [f"{i.code}:{i.column}" for i in issues if i.severity == "error"]
        for f in fields.values():
            if f.required and rec[f.name] is None and f.name not in allow_missing and not any(r.endswith(f":{f.name}") for r in reasons):
                reasons.append(f"missing_required:{f.name}")
        out.issues.extend(issues)
        if reasons:
            out.quarantined.append((rno, ",".join(reasons)))
            continue
        k = rec[key_field]
        rec["_row"] = rno
        if k in seen:
            if record_hash(seen[k]) == record_hash(rec):
                n_ident += 1
            else:
                n_conf += 1
        seen[k] = rec
    out.records = list(seen.values())
    out.stats = {"rows_in": len(table.rows), "loaded": len(out.records), "quarantined": len(out.quarantined),
                 "duplicates_identical": n_ident, "duplicates_conflicting": n_conf,
                 "dropped_junk": sum(table.meta.get("dropped", {}).values()), "unmapped_required": sorted(
                     f.name for f in fields.values() if f.required and f.name not in mapped_targets and f.name not in allow_missing)}
    return out
