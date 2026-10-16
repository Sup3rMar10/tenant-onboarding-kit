"""Schema drift: has a later export changed shape since the config was built?

The config keeps a fingerprint of every column it was built from (type, shape, hashed
value sketch). A new export is read the same way and compared:

  dropped      a column in the config is gone
  added        a column the config has never seen
  renamed      a dropped and an added column whose contents match (same kind, overlapping values)
  retyped      same name, different content: the format changed (date format, decimal
               separator), or the type itself changed (an amount that is suddenly whole numbers, an
               id whose shape changed)

and each kind has a policy: heal (update the config and carry on, and say so), alert
(load, but tell a person), or block (do not load). The policy table lives in the config so
a customer's risk appetite is explicit and reviewable.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .config import DEFAULT_POLICY, DEFAULT_THRESHOLDS
from .mapper import column_header_scores, value_fit
from .profile import Profile, profile_table, sketch_overlap
from .readers import RawTable
from .schema import ENTITIES, get_field

FAMILY = {"text": "cat", "status": "cat", "method": "cat", "country": "cat", "currency": "cat", "id": "idint", "int": "idint"}


def family(kind: str) -> str:
    return FAMILY.get(kind, kind)


@dataclass
class DriftEvent:
    kind: str
    file: str
    column: str
    new_column: str | None = None
    target: str | None = None
    entity: str = ""
    confidence: float = 1.0
    action: str = "alert"
    detail: str = ""
    extra: dict = field(default_factory=dict)

    def key(self) -> tuple[str, str, str]:
        return (self.kind, self.file, self.column)


@dataclass
class DriftParams:
    rename_threshold: float = DEFAULT_THRESHOLDS["rename_confidence"]


def similarity(fp: dict, p: Profile, old: str, new: str) -> float:
    if fp["kind"] == p.kind:
        kind = 1.0
    elif family(fp["kind"]) == family(p.kind):
        kind = 0.8
    else:
        kind = 0.0
    sk = sketch_overlap(fp.get("sketch", []), p.sketch)
    shape = 1.0 if fp.get("shape_top") == p.shape_top else 0.0
    if fp.get("num_min") is not None and p.num_min is not None:
        lo, hi = max(fp["num_min"], p.num_min), min(fp["num_max"], p.num_max)
        union = max(fp["num_max"], p.num_max) - min(fp["num_min"], p.num_min)
        stat = max(0.0, hi - lo) / union if union > 0 else 1.0
    else:
        a, b = fp.get("len_mean", 0.0), p.len_mean
        stat = 1 - min(1.0, abs(a - b) / max(a, b, 1.0))
    name = SequenceMatcher(None, old.lower(), new.lower()).ratio()
    return 0.30 * kind + 0.30 * sk + 0.10 * shape + 0.15 * stat + 0.15 * name


def _retype(fp: dict, p: Profile) -> tuple[str, str, dict] | None:
    """Same column name in both: did its content change? Returns (kind, detail, extra) or None."""
    if fp["kind"] == "empty":
        return None
    if p.kind == "empty":
        return "retyped_incompatible", "column is now empty", {}
    if family(fp["kind"]) != family(p.kind):
        return "retyped_incompatible", f"content changed from {fp['kind']} to {p.kind}", {}
    if fp["kind"] == "date" and p.kind == "date":
        if not set(fp.get("date_candidates") or [fp.get("date_format")]) & set(p.date_candidates):
            return "retyped_format", f"date format {fp.get('date_format')} -> {p.date_format}", {"format": p.date_format}
        return None
    if fp["kind"] == "money" and p.kind == "money" and fp.get("decimal") and p.decimal != fp["decimal"] and p.money_decimals >= 0.3:
        return "retyped_format", f"decimal separator '{fp['decimal']}' -> '{p.decimal}'", {"decimal": p.decimal}
    if fp["kind"] == "id" and p.kind == "id" and fp.get("shape_share", 0) >= 0.7 and p.shape_share >= 0.7:
        longer = max(fp.get("len_mean", 0.0), p.len_mean, 1.0)
        if fp.get("shape_top") != p.shape_top or abs(fp.get("len_mean", 0.0) - p.len_mean) / longer > 0.2:
            return "retyped_incompatible", f"id changed from {fp.get('shape_top')} (length ~{fp.get('len_mean', 0):.0f}) to {p.shape_top} (length ~{p.len_mean:.0f})", {}
    return None


def detect_drift(fcfg: dict, table: RawTable, policy: dict | None = None, params: DriftParams | None = None,
                 profiles: list[Profile] | None = None) -> list[DriftEvent]:
    policy = {**DEFAULT_POLICY, **(policy or {})}
    params = params or DriftParams()
    entity = fcfg["entity"]
    base = fcfg["baseline"]
    target_of = {c["source"]: c.get("target") for c in fcfg["columns"]}
    profs = dict(zip(table.columns, profiles if profiles is not None else profile_table(table)))
    dropped = [c for c in base if c not in profs]
    added = [c for c in table.columns if c not in base]
    events: list[DriftEvent] = []

    pairs = sorted(((similarity(base[d], profs[a], d, a), d, a) for d in dropped for a in added), reverse=True)
    used_d: set[str] = set()
    used_a: set[str] = set()
    for sim, d, a in pairs:
        if sim < params.rename_threshold or d in used_d or a in used_a:
            continue
        used_d.add(d)
        used_a.add(a)
        detail, extra = f"'{d}' looks like '{a}' (similarity {sim:.2f})", {}
        r = _retype(base[d], profs[a])
        if r and r[0] == "retyped_format":                    # renamed and reformatted in the same export
            detail, extra = detail + "; " + r[1], r[2]
        events.append(DriftEvent("renamed", table.name, d, a, target_of.get(d), entity, round(sim, 3), policy["renamed"], detail, extra))
    for d in dropped:
        if d in used_d:
            continue
        tgt = target_of.get(d)
        required = bool(tgt) and get_field(entity, tgt).required
        kind = "dropped_required" if required else "dropped_optional"
        events.append(DriftEvent(kind, table.name, d, None, tgt, entity, 1.0, policy[kind],
                                 f"column '{d}' is gone" + (f" (maps to required field {tgt})" if required else "")))
    mapped_targets = {t for t in target_of.values() if t}
    for a in added:
        if a in used_a:
            continue
        p = profs[a]
        hs = column_header_scores(a)
        best, best_s = None, 0.0
        for f in ENTITIES[entity]:
            if f.name in mapped_targets:
                continue
            s = 0.5 * hs[f"{entity}.{f.name}"] + 0.5 * value_fit(p, f)
            if s > best_s:
                best, best_s = f.name, s
        sugg = f"; could be {best} (score {best_s:.2f})" if best and best_s >= 0.6 else ""
        events.append(DriftEvent("added", table.name, a, None, best if sugg else None, entity, 1.0, policy["added"], f"new column '{a}'{sugg}"))
    for col, fp in base.items():
        if col not in profs:
            continue
        r = _retype(fp, profs[col])
        if r:
            kind, detail, extra = r
            events.append(DriftEvent(kind, table.name, col, None, target_of.get(col), entity, 1.0, policy[kind], detail, extra))
    return events


def apply_heals(fcfg: dict, events: list[DriftEvent]) -> dict:
    """A copy of the file config with every `heal` event applied. Healed columns are marked for review."""
    out = copy.deepcopy(fcfg)
    for e in events:
        if e.action != "heal":
            continue
        col = next((c for c in out["columns"] if c["source"] == e.column), None)
        if col is None:
            continue
        if e.kind == "renamed" and e.new_column:
            col["source"] = e.new_column
            col["needs_review"] = True
            col["healed"] = f"renamed from {e.column}"
            if e.extra:
                col.setdefault("transform", {}).update(e.extra)
            if e.column in out["baseline"]:
                out["baseline"][e.new_column] = out["baseline"].pop(e.column)
        elif e.kind == "retyped_format":
            tr = col.setdefault("transform", {})
            tr.update({k: v for k, v in e.extra.items()})
            col["needs_review"] = True
            col["healed"] = e.detail
    return out


def decision(events: list[DriftEvent]) -> str:
    acts = {e.action for e in events}
    return "block" if "block" in acts else "alert" if "alert" in acts else "heal" if "heal" in acts else "clean"
