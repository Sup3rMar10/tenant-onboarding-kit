"""Scoring: the tuning/held-out split, mapping precision and recall, bootstrap intervals.

The split is made once, from a fixed seed, stratified by difficulty. Tuning code receives
only the tuning tenants. The held-out tenants' truth is read only by the final evaluation.
"""
from __future__ import annotations

import hashlib
import random
from collections import defaultdict

from .generate import Tenant
from .mapper import TenantMapping

SPLIT_SEED = 20261016


def split_tenants(tenants: list[Tenant], seed: int = SPLIT_SEED) -> tuple[list[Tenant], list[Tenant]]:
    by_diff: dict[str, list[Tenant]] = defaultdict(list)
    for t in sorted(tenants, key=lambda t: t.tid):
        by_diff[t.difficulty].append(t)
    tuning: list[Tenant] = []
    held: list[Tenant] = []
    for _, ts in sorted(by_diff.items()):
        ts = sorted(ts, key=lambda t: hashlib.sha256(f"{seed}:{t.tid}".encode()).hexdigest())
        k = (len(ts) + 1) // 2
        tuning += ts[:k]
        held += ts[k:]
    return sorted(tuning, key=lambda t: t.tid), sorted(held, key=lambda t: t.tid)


def truth_set(truth: dict) -> set[tuple[str, str, str, str]]:
    return {(fn, src, meta["entity"], target.split(".", 1)[1])
            for fn, meta in truth["files"].items() for src, target in meta["columns"].items() if target}


def score_mapping(mapping: TenantMapping, truth: dict) -> dict:
    ts = truth_set(truth)
    pred = mapping.predictions()
    n_cols = sum(len(m["columns"]) for m in truth["files"].values())
    correct_cols = 0
    for fn, meta in truth["files"].items():
        fm = mapping.files.get(fn)
        got = fm.by_source() if fm else {}
        for src, target in meta["columns"].items():
            want = target.split(".", 1)[1] if target else None
            cm = got.get(src)
            have = cm.field if cm else None
            if have == want and (want is None or (fm.entity == meta["entity"])):
                correct_cols += 1
    ent_ok = sum(1 for fn, meta in truth["files"].items() if fn in mapping.files and mapping.files[fn].entity == meta["entity"])
    return {"tp": len(pred & ts), "pred": len(pred), "truth": len(ts), "files": len(truth["files"]), "entity_ok": ent_ok,
            "cols": n_cols, "cols_ok": correct_cols,
            "wrong": sorted(pred - ts), "missed": sorted(ts - pred)}


def _prf(tp: int, pred: int, truth: int) -> tuple[float, float, float]:
    p = tp / pred if pred else 0.0
    r = tp / truth if truth else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def aggregate(scores: list[dict]) -> dict:
    tp, pred, truth = (sum(s[k] for s in scores) for k in ("tp", "pred", "truth"))
    p, r, f = _prf(tp, pred, truth)
    files, ent = sum(s["files"] for s in scores), sum(s["entity_ok"] for s in scores)
    cols, ok = sum(s["cols"] for s in scores), sum(s["cols_ok"] for s in scores)
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "entity_accuracy": round(ent / files, 4) if files else 0.0,
            "column_accuracy": round(ok / cols, 4) if cols else 0.0, "tenants": len(scores), "columns_mapped": pred, "columns_true": truth}


def mean_f1(scores: list[dict]) -> float:
    fs = [_prf(s["tp"], s["pred"], s["truth"])[2] for s in scores]
    return sum(fs) / len(fs) if fs else 0.0


def bootstrap_ci(scores: list[dict], stat=lambda sc: aggregate(sc)["f1"], n: int = 1000, seed: int = 11) -> tuple[float, float]:
    """95% interval for a statistic, resampling whole tenants."""
    rng = random.Random(seed)
    vals = sorted(stat([rng.choice(scores) for _ in scores]) for _ in range(n))
    return round(vals[int(0.025 * n)], 4), round(vals[int(0.975 * n)], 4)


def by_difficulty(tenants: list[Tenant], scores: list[dict]) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for t, s in zip(tenants, scores):
        groups[t.difficulty].append(s)
    return {d: aggregate(v) for d, v in sorted(groups.items())}


# ------------------------------------------------------------ normalization scoring


def oracle_mapping(truth: dict) -> TenantMapping:
    """The mapping a perfect mapper would produce. Used to measure the transform layer on its own."""
    from .mapper import ColumnMapping, FileMapping

    files = {}
    for fn, meta in truth["files"].items():
        cols = [ColumnMapping(src, tgt.split(".", 1)[1] if tgt else None, 1.0 if tgt else 0.0) for src, tgt in meta["columns"].items()]
        files[fn] = FileMapping(fn, meta["entity"], {}, cols)
    return TenantMapping(files)


def score_normalization(normalized: dict, truth: dict) -> dict:
    """Compare normalized records (by entity) with the canonical truth records."""
    from .schema import ENTITIES

    out = {"truth_records": 0, "found": 0, "fields": 0, "fields_ok": 0, "false_loads": 0, "quarantine_expected": 0,
           "quarantine_held_back": 0, "field_errors": {}}
    for entity, recs in truth["records"].items():
        key = f"{entity}_id"
        names = [f.name for f in ENTITIES[entity]]
        got = {}
        for nf in normalized.values():
            if nf.entity == entity:
                got.update({r[key]: r for r in nf.records})
        want = {r[key]: r for r in recs}
        out["truth_records"] += len(want)
        out["false_loads"] += sum(1 for k in got if k not in want)
        for k, w in want.items():
            g = got.get(k)
            if g is None:
                continue
            out["found"] += 1
            for n in names:
                out["fields"] += 1
                if g.get(n) == w.get(n):
                    out["fields_ok"] += 1
                else:
                    out["field_errors"][f"{entity}.{n}"] = out["field_errors"].get(f"{entity}.{n}", 0) + 1
        for k in truth["expected_quarantine"].get(entity, []):
            out["quarantine_expected"] += 1
            out["quarantine_held_back"] += int(k not in got)
    return out


def aggregate_normalization(scores: list[dict]) -> dict:
    tr = sum(s["truth_records"] for s in scores)
    found = sum(s["found"] for s in scores)
    fields = sum(s["fields"] for s in scores)
    ok = sum(s["fields_ok"] for s in scores)
    errs: dict[str, int] = {}
    for s in scores:
        for k, v in s["field_errors"].items():
            errs[k] = errs.get(k, 0) + v
    qe = sum(s["quarantine_expected"] for s in scores)
    return {"record_recall": round(found / tr, 4) if tr else 0.0, "field_accuracy": round(ok / fields, 4) if fields else 0.0,
            "false_loads": sum(s["false_loads"] for s in scores),
            "junk_rows_held_back": round(sum(s["quarantine_held_back"] for s in scores) / qe, 4) if qe else 1.0,
            "field_errors": dict(sorted(errs.items(), key=lambda kv: -kv[1]))}
