"""Per-tenant configuration as code.

The mapper's output is turned into a plain JSON file: one entry per source file with
its format, and one entry per source column with the target field, how confident the
mapper was, whether a person should look at it, and the transform to apply (date
format, decimal separator, status vocabulary...). The file is meant to be reviewed,
edited and committed. `load_config` validates it and says what is wrong in words.

It also stores a fingerprint of every column (hashed value sketch, shape, type) so a
later export can be compared against what the mapping was built from (drift.py).
"""
from __future__ import annotations

import json
from pathlib import Path

from . import __version__
from .mapper import TenantMapping, TenantSignals
from .parsing import DATE_FORMATS
from .profile import Profile
from .readers import RawTable
from .schema import ENTITIES, METHOD_LOOKUP, STATUS_LOOKUP, get_field, header_tokens, norm_text

CONFIG_VERSION = 1
CENTS_TOKENS = {"cents", "cent", "centavos", "centimos", "minor", "pence"}
DEFAULT_POLICY = {"renamed": "heal", "added": "alert", "dropped_optional": "alert", "dropped_required": "block",
                  "retyped_format": "heal", "retyped_incompatible": "block"}
DEFAULT_THRESHOLDS = {"max_quarantine_rate": 0.25, "max_orphan_rate": 0.3, "rename_confidence": 0.7}
ACTIONS = {"heal", "alert", "block"}


class ConfigError(ValueError):
    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


def _transform(field_type: str, source: str, p: Profile) -> dict:
    if field_type == "date":
        return {"type": "date", "format": p.date_format or "%Y-%m-%d", "ambiguous": p.date_ambiguous}
    if field_type == "money":
        cents = bool(set(header_tokens(source.split(".")[-1])) & CENTS_TOKENS)
        return {"type": "money", "decimal": p.decimal, "scale": 0.01 if cents else 1, "currency_in_value": p.currency_token >= 0.5}
    if field_type in ("status", "method"):
        lookup = STATUS_LOOKUP if field_type == "status" else METHOD_LOOKUP
        known = {v: lookup[norm_text(v)] for v in sorted(p.values) if norm_text(v) in lookup}
        unknown = sorted(v for v in p.values if norm_text(v) not in lookup)
        return {"type": field_type, "value_map": known, "unmapped": unknown}
    return {"type": field_type}


def build_config(tenant_id: str, tables: list[RawTable], mapping: TenantMapping, signals: TenantSignals) -> dict:
    files = {}
    for t, fs in zip(tables, signals.files):
        fm = mapping.files[t.name]
        cols = []
        mapped = set()
        by_source = fm.by_source()
        for cs in fs.columns:                       # file order; looked up by name, because a mapping may list columns in any order
            c = by_source[cs.source]
            entry: dict = {"source": c.source, "target": c.field}
            if c.field:
                ftype = get_field(fm.entity, c.field).ftype
                entry.update({"confidence": c.score, "needs_review": c.review, "decided_by": c.decided_by,
                              "transform": _transform(ftype, c.source, cs.profile)})
                mapped.add(c.field)
            else:
                entry["suggestions"] = [{"target": k, "score": s} for k, s in c.alternatives[:2] if s >= 0.2]
            cols.append(entry)
        missing = [f.name for f in ENTITIES[fm.entity] if f.required and f.name not in mapped]
        if fm.entity == "invoice" and "currency" in missing:
            amount = next((c for c in cols if c.get("target") == "amount"), None)
            if amount and amount["transform"].get("currency_in_value"):
                missing.remove("currency")
        meta = t.meta
        files[t.name] = {
            "entity": fm.entity,
            "format": {"kind": meta.get("kind"), "encoding": meta.get("encoding"), "delimiter": meta.get("delimiter")},
            "columns": cols,
            "required_unmapped": missing,
            "baseline": {cs.source: cs.profile.fingerprint() for cs in fs.columns},
        }
    return {"config_version": CONFIG_VERSION, "generated_by": f"tenantkit {__version__}", "tenant_id": tenant_id,
            "drift_policy": dict(DEFAULT_POLICY), "thresholds": dict(DEFAULT_THRESHOLDS), "files": files}


def validate_config(cfg: dict) -> list[str]:
    problems = []
    if cfg.get("config_version") != CONFIG_VERSION:
        problems.append(f"config_version must be {CONFIG_VERSION}")
    for kind, action in cfg.get("drift_policy", {}).items():
        if kind not in DEFAULT_POLICY:
            problems.append(f"drift_policy: unknown drift kind '{kind}'")
        elif action not in ACTIONS:
            problems.append(f"drift_policy.{kind}: '{action}' is not one of {sorted(ACTIONS)}")
    for fname, fc in cfg.get("files", {}).items():
        ent = fc.get("entity")
        if ent not in ENTITIES:
            problems.append(f"{fname}: unknown entity '{ent}'")
            continue
        valid = {f.name for f in ENTITIES[ent]}
        seen: dict[str, str] = {}
        for col in fc.get("columns", []):
            tgt = col.get("target")
            if tgt is None:
                continue
            if tgt not in valid:
                problems.append(f"{fname}: column '{col['source']}' maps to '{tgt}', which is not a {ent} field")
                continue
            if tgt in seen:
                problems.append(f"{fname}: '{tgt}' is mapped from both '{seen[tgt]}' and '{col['source']}'")
            seen[tgt] = col["source"]
            tr = col.get("transform", {})
            if tr.get("type") == "date" and tr.get("format") not in DATE_FORMATS:
                problems.append(f"{fname}: column '{col['source']}' has unknown date format '{tr.get('format')}'")
            if tr.get("type") == "money" and tr.get("decimal") not in (".", ","):
                problems.append(f"{fname}: column '{col['source']}' decimal separator must be '.' or ','")
        for f in ENTITIES[ent]:
            if f.required and f.name not in seen and f.name not in fc.get("allow_missing", []):
                if not (ent == "invoice" and f.name == "currency" and _currency_embedded(fc)):
                    problems.append(f"{fname}: required field '{f.name}' is not mapped (map it, or list it under allow_missing)")
    return problems


def _currency_embedded(fc: dict) -> bool:
    return any(c.get("target") == "amount" and c.get("transform", {}).get("currency_in_value") for c in fc.get("columns", []))


def save_config(cfg: dict, path: Path | str) -> None:
    Path(path).write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_config(path: Path | str) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = validate_config(cfg)
    if problems:
        raise ConfigError(problems)
    return cfg
