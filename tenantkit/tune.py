"""Choose the mapper's settings, and the drift detector's rename threshold, on the tuning tenants only.

Mirrors the discipline of the sibling project rag-tuning-lab:

  * the functions here take the tuning tenants and nothing else; there is no argument through which a
    held-out tenant could reach them (a test poisons the held-out truth and checks the choice cannot move);
  * one setting changes at a time, the best value is kept, and two passes are made;
  * among settings within 0.003 F1 of the best, the one that sends the fewest columns to human review
    wins, because review time is the real cost;
  * every configuration tried is appended to results/trials.jsonl so the search can be audited.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from .drift import DriftParams, detect_drift
from .evaluate import aggregate, score_mapping, truth_set
from .generate import Tenant
from .mapper import Params, TenantSignals, compute_signals, map_tenant
from .profile import profile_table
from .readers import RawTable, read_table

TOLERANCE = 0.003

# The starting point is plain header + value matching with every extra switched off.
START = Params(w_header=1.0, w_value=1.0, w_rel=0.0, w_file=0.0, threshold=0.5, header_prior=0.55, one_per_entity=False,
               ignore_factor=1.0, date_rule=False, magnitude_rule=False, residual_rule=False)

STEPS: list[tuple[str, list]] = [
    ("w_header", [0.25, 0.5, 1.0, 2.0]),
    ("w_rel", [0.0, 0.25, 0.5, 1.0]),
    ("threshold", [0.35, 0.45, 0.55, 0.65]),
    ("header_prior", [0.3, 0.55, 0.8]),
    ("w_file", [0.0, 0.5, 1.0]),
    ("one_per_entity", [False, True]),
    ("ignore_factor", [1.0, 0.6]),
    ("date_rule", [False, True]),
    ("magnitude_rule", [False, True]),
    ("residual_rule", [False, True]),
]


def load_signals(t: Tenant, sync: str = "base") -> TenantSignals:
    return compute_signals([read_table(p) for p in t.files(sync)])


def evaluate_params(params: Params, tenants: list[Tenant], signals: dict[str, TenantSignals]) -> tuple[dict, float]:
    scores = []
    flagged = mapped = 0
    for t in tenants:
        m = map_tenant(signals[t.tid], params)
        scores.append(score_mapping(m, t.truth()))
        for fm in m.files.values():
            for c in fm.columns:
                if c.field:
                    mapped += 1
                    flagged += int(c.review)
    return aggregate(scores), (flagged / mapped if mapped else 0.0)


def tune_mapper(tuning: list[Tenant], trials_path: Path | None = None, passes: int = 2) -> tuple[Params, list[dict]]:
    signals = {t.tid: load_signals(t) for t in tuning}
    cur = START
    trials: list[dict] = []
    best_agg, best_review = evaluate_params(cur, tuning, signals)
    trials.append({"stage": "start", "params": cur.to_dict(), **best_agg, "review_rate": round(best_review, 4)})
    for p in range(passes):
        for name, values in STEPS:
            results = []
            for v in values:
                cand = replace(cur, **{name: v})
                agg, rv = evaluate_params(cand, tuning, signals)
                results.append((agg["f1"], rv, v, cand))
                trials.append({"stage": f"pass{p + 1}:{name}", "params": cand.to_dict(), **agg, "review_rate": round(rv, 4)})
            top = max(r[0] for r in results)
            near = [r for r in results if top - r[0] <= TOLERANCE]
            # fewest flagged for review, then keep the current value if it is among the contenders (stability)
            near.sort(key=lambda r: (round(r[1], 4), r[2] != getattr(cur, name)))
            cur = near[0][3]
    if trials_path:
        with open(trials_path, "w", encoding="utf-8") as fh:
            for tr in trials:
                fh.write(json.dumps(tr) + "\n")
    return cur, trials


# ------------------------------------------------------------------ drift threshold

RENAME_GRID = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80]


def drift_truth_keys(events: list[dict]) -> set[tuple[str, str, str]]:
    kind_map = {"rename_mild": "renamed", "rename_major": "renamed", "add_unmapped": "added", "add_mappable": "added",
                "drop_optional": "dropped_optional", "drop_required": "dropped_required", "retype_date": "retyped_format",
                "retype_decimal": "retyped_format", "retype_cents": "retyped_incompatible", "id_shape": "retyped_incompatible"}
    return {(kind_map[e["kind"]], e["file"], e["column"]) for e in events}


_TABLE_CACHE: dict[tuple[str, str, str], tuple] = {}


def sync_tables(t: Tenant, sync: str) -> list[tuple[RawTable, list]]:
    """Read and profile a tenant's sync once; the same export is compared at many thresholds."""
    out = []
    for p in t.files(sync):
        key = (str(t.root), sync, p.name)
        if key not in _TABLE_CACHE:
            tb = read_table(p)
            _TABLE_CACHE[key] = (tb, profile_table(tb))
        out.append(_TABLE_CACHE[key])
    return out


def detect_for_tenant(cfg: dict, t: Tenant, sync: str, tau: float) -> list:
    out = []
    for tb, profs in sync_tables(t, sync):
        if tb.name in cfg["files"]:
            out += detect_drift(cfg["files"][tb.name], tb, cfg["drift_policy"], DriftParams(tau), profs)
    return out


def tune_drift(tuning: list[Tenant], configs: dict[str, dict]) -> tuple[float, list[dict]]:
    """Pick the rename threshold by event F1 over the tuning tenants' drift and control syncs."""
    trials = []
    for tau in RENAME_GRID:
        tp = fp = fn = 0
        for t in tuning:
            truth = drift_truth_keys(t.truth("drift")["events"])
            found = {e.key() for e in detect_for_tenant(configs[t.tid], t, "drift", tau)}
            ctrl = {e.key() for e in detect_for_tenant(configs[t.tid], t, "control", tau)}
            tp += len(found & truth)
            fp += len(found - truth) + len(ctrl)
            fn += len(truth - found)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        trials.append({"stage": "drift:rename_threshold", "rename_threshold": tau, "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)})
    top = max(tr["f1"] for tr in trials)
    near = [tr for tr in trials if top - tr["f1"] <= TOLERANCE]
    best = max(near, key=lambda tr: tr["rename_threshold"])          # near-tie: the stricter threshold heals fewer things by mistake
    return best["rename_threshold"], trials
