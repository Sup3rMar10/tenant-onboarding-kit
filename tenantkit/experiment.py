"""Run the whole evaluation and write results/.

Order matters and is the point:

  1. split the tenants (fixed seed, stratified by difficulty);
  2. tune the mapper and the drift threshold on the tuning tenants only;
  3. score everything once on the held-out tenants: mapping against baselines and ablations,
     normalization, drift detection, and the ingest guarantees.

`dev=True` stops after scoring the tuning tenants, which is what is used while changing code.
"""
from __future__ import annotations

import json
import platform
import random
import shutil
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from . import __version__
from .config import build_config, DEFAULT_POLICY
from .drift import DriftParams, decision
from .evaluate import (aggregate, aggregate_normalization, bootstrap_ci, by_difficulty, oracle_mapping, score_mapping,
                       score_normalization, split_tenants, SPLIT_SEED)
from .generate import GEN_SEED, Tenant, generate_all, lexicon_coverage, load_tenants
from .ingest import InjectedFailure, Ledger, RollbackError, ingest_tenant, rollback
from .mapper import Params, TenantSignals, compute_signals, map_tenant, naive_map
from .normalize import normalize_table
from .readers import read_table
from .tune import (RENAME_GRID, detect_for_tenant, drift_truth_keys, evaluate_params, load_signals, tune_drift, tune_mapper)

THRESHOLD_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
BLOCKING = {"dropped_required", "retyped_incompatible"}
ALERTING = {"added", "dropped_optional"}


# ------------------------------------------------------------------ mapping


def _scores(system, tenants: list[Tenant], signals: dict, tables: dict) -> list[dict]:
    out = []
    for t in tenants:
        m = naive_map(tables[t.tid]) if system == "naive" else map_tenant(signals[t.tid], system)
        out.append(score_mapping(m, t.truth()))
    return out


def _review_stats(system: Params, tenants, signals) -> dict:
    """If a person reviews only the flagged columns, how good are the columns they skip?"""
    flagged = kept = wrong_flagged = wrong_kept = 0
    for t in tenants:
        m = map_tenant(signals[t.tid], system)
        truth = {(fn, src, meta["entity"], tgt.split(".", 1)[1]) for fn, meta in t.truth()["files"].items() for src, tgt in meta["columns"].items() if tgt}
        for fn, fm in m.files.items():
            for c in fm.columns:
                if not c.field:
                    continue
                ok = (fn, c.source, fm.entity, c.field) in truth
                if c.review:
                    flagged += 1
                    wrong_flagged += not ok
                else:
                    kept += 1
                    wrong_kept += not ok
    total = flagged + kept
    wrong = wrong_flagged + wrong_kept
    return {"columns_mapped": total, "flagged_for_review": flagged, "flagged_share": round(flagged / total, 4) if total else 0.0,
            "wrong_predictions": wrong, "wrong_caught_by_flags": round(wrong_flagged / wrong, 4) if wrong else 1.0,
            "precision_of_unflagged": round(1 - wrong_kept / kept, 4) if kept else 0.0}


def _tune_threshold(template: Params, tuning, signals) -> Params:
    best = None
    for th in THRESHOLD_GRID:
        p = replace(template, threshold=th)
        f1 = evaluate_params(p, tuning, signals)[0]["f1"]
        if best is None or f1 > best[0] + 1e-9:
            best = (f1, p)
    return best[1]


def _paired_ci(a: list[dict], b: list[dict], n: int = 1000, seed: int = 5) -> tuple[float, float]:
    """95% interval for F1(a) - F1(b), resampling whole tenants."""
    rng = random.Random(seed)
    idx = range(len(a))
    vals = []
    for _ in range(n):
        pick = [rng.choice(idx) for _ in idx]
        vals.append(aggregate([a[i] for i in pick])["f1"] - aggregate([b[i] for i in pick])["f1"])
    vals.sort()
    return round(vals[int(0.025 * n)], 4), round(vals[int(0.975 * n)], 4)


def baseline_systems(tuned: Params, tuning, signals) -> dict[str, Params]:
    off = dict(w_rel=0.0, w_file=0.0, one_per_entity=False, ignore_factor=1.0, date_rule=False, magnitude_rule=False, residual_rule=False)
    header_only = _tune_threshold(replace(tuned, w_header=1.0, w_value=0.0, header_prior=0.0, **off), tuning, signals)
    value_only = _tune_threshold(replace(tuned, w_header=0.0, w_value=1.0, **off), tuning, signals)
    header_value = _tune_threshold(replace(tuned, w_header=1.0, w_value=1.0, **off), tuning, signals)
    return {"header only": header_only, "value only": value_only, "header + value": header_value}


def ablations(tuned: Params) -> dict[str, Params]:
    out = {"no header signal": replace(tuned, w_header=0.0), "no value signal": replace(tuned, w_value=0.0),
           "no relation signal": replace(tuned, w_rel=0.0), "no file-name hint": replace(tuned, w_file=0.0),
           "no one-file-per-entity": replace(tuned, one_per_entity=False), "no date rule": replace(tuned, date_rule=False),
           "no magnitude rule": replace(tuned, magnitude_rule=False), "no residual rule": replace(tuned, residual_rule=False),
           "no ignore tokens": replace(tuned, ignore_factor=1.0)}
    return {k: v for k, v in out.items() if v != tuned}


# ------------------------------------------------------------------ configs


def predicted_config(t: Tenant, params: Params, sig: TenantSignals | None = None) -> tuple[dict, list]:
    tables = [read_table(p) for p in t.files()]
    sig = sig or compute_signals(tables)
    return build_config(t.tid, tables, map_tenant(sig, params), sig), tables


def oracle_config(t: Tenant) -> tuple[dict, list]:
    tables = [read_table(p) for p in t.files()]
    sig = compute_signals(tables)
    return build_config(t.tid, tables, oracle_mapping(t.truth()), sig), tables


# ------------------------------------------------------------------ normalization


def eval_normalization(tenants: list[Tenant], cfgs: dict[str, tuple[dict, list]], sync: str = "base") -> dict:
    scores = []
    for t in tenants:
        cfg, _ = cfgs[t.tid]
        tables = [read_table(p) for p in t.files(sync)]
        norm = {tb.name: normalize_table(tb, cfg["files"][tb.name]) for tb in tables if tb.name in cfg["files"]}
        scores.append(score_normalization(norm, t.truth(sync)))
    return aggregate_normalization(scores)


# ------------------------------------------------------------------ drift


def _expected_decision(keys: set) -> str:
    kinds = {k[0] for k in keys}
    return "block" if kinds & BLOCKING else "alert" if kinds & ALERTING else "heal" if kinds else "clean"


def eval_drift(tenants: list[Tenant], cfgs: dict[str, tuple[dict, list]], tau: float) -> dict:
    tp = fp = fn = 0
    per_kind: dict[str, list[int]] = {}
    exp_ok = unsafe = n_block_expected = 0
    ctrl_events = ctrl_any = ctrl_block = 0
    avoided = {"fields_wrong": 0, "fields": 0, "syncs": 0}
    for t in tenants:
        cfg = cfgs[t.tid][0]
        truth = drift_truth_keys(t.truth("drift")["events"])
        ev = detect_for_tenant(cfg, t, "drift", tau)
        found = {e.key() for e in ev}
        tp += len(found & truth)
        fp += len(found - truth)
        fn += len(truth - found)
        for k in truth:
            per_kind.setdefault(k[0], [0, 0])
            per_kind[k[0]][1] += 1
            per_kind[k[0]][0] += int(k in found)
        exp = _expected_decision(truth)
        got = decision(ev)
        exp_ok += int(exp == got)
        if exp == "block":
            n_block_expected += 1
            unsafe += int(got != "block")
        if got == "block" and exp == "block":
            # what would an ungated load of this export have written? (oracle transforms from the base export)
            bad_tables = [read_table(p) for p in t.files("drift")]
            norm = {tb.name: normalize_table(tb, cfg["files"][tb.name]) for tb in bad_tables if tb.name in cfg["files"]}
            s = score_normalization(norm, t.truth("drift"))
            avoided["fields_wrong"] += s["fields"] - s["fields_ok"] + 0
            avoided["fields"] += s["fields"]
            avoided["syncs"] += 1
        cev = detect_for_tenant(cfg, t, "control", tau)
        ctrl_events += len(cev)
        ctrl_any += int(bool(cev))
        ctrl_block += int(decision(cev) == "block")
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"events_true": tp + fn, "precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
            "recall_by_kind": {k: {"found": a, "of": b} for k, (a, b) in sorted(per_kind.items())},
            "sync_decision_accuracy": round(exp_ok / len(tenants), 4), "syncs": len(tenants),
            "unsafe_misses": unsafe, "should_block": n_block_expected,
            "control_syncs": len(tenants), "control_false_alarms": ctrl_events, "control_syncs_with_alarm": ctrl_any,
            "control_false_blocks": ctrl_block, "ungated_load_would_corrupt": avoided}


# ------------------------------------------------------------------ ingest guarantees


def _ledger_with(cfg, base_files, sync_files, **kw):
    led = Ledger()
    r1 = ingest_tenant(led, cfg, base_files)
    r2 = ingest_tenant(led, cfg, sync_files, **kw) if sync_files else None
    return led, r1, r2


def ingest_checks(t: Tenant, cfg: dict) -> list[tuple[str, bool, str]]:
    tid = cfg["tenant_id"]
    base, ctrl = t.files("base"), t.files("control")
    out: list[tuple[str, bool, str, bool]] = []

    def check(name, ok, detail="", exercised=True):
        """`exercised` is False when the scenario never actually happened (nothing to crash, nothing to roll back)."""
        out.append((name, bool(ok), detail, exercised))

    # 1. idempotent: the same export twice changes nothing
    led = Ledger()
    ingest_tenant(led, cfg, base)
    before = led.snapshot(tid)
    again = ingest_tenant(led, cfg, base)
    check("idempotent", again.status == "noop" and led.snapshot(tid) == before, f"second run status {again.status}", len(before) > 0)

    # 2. incremental: after base then a fuller export, every row of that export is present with its content
    led_inc = Ledger()
    ingest_tenant(led_inc, cfg, base)
    r = ingest_tenant(led_inc, cfg, ctrl)
    led_full = Ledger()
    ingest_tenant(led_full, cfg, ctrl)
    full, inc = led_full.snapshot(tid), led_inc.snapshot(tid)
    check("incremental", r.status in ("committed", "noop") and all(inc.get(k) == v for k, v in full.items()), f"status {r.status}", r.status == "committed")

    # 3. dry run writes nothing and predicts the real run's counts
    led_dry = Ledger()
    ingest_tenant(led_dry, cfg, base)
    snap, nruns = led_dry.snapshot(tid), len(led_dry.runs(tid))
    dry = ingest_tenant(led_dry, cfg, ctrl, dry_run=True)
    untouched = led_dry.snapshot(tid) == snap and len(led_dry.runs(tid)) == nruns
    real = ingest_tenant(led_dry, cfg, ctrl)
    key = lambda res: sorted((f["file"], f.get("inserted", 0), f.get("updated", 0)) for f in res.files)
    check("dry_run", untouched and key(dry) == key(real), f"untouched={untouched}", any(f.get("inserted") or f.get("updated") for f in dry.files))

    # 4. a crash between files, then resume, ends in the same state as an uninterrupted load
    led_c = Ledger()
    ingest_tenant(led_c, cfg, base)
    crashed = False
    try:
        ingest_tenant(led_c, cfg, ctrl, fail_after=1)
    except InjectedFailure:
        crashed = True
    failed = [r for r in led_c.runs(tid) if r["status"] == "failed"]
    if crashed and failed:
        ingest_tenant(led_c, cfg, ctrl, resume_run=failed[-1]["run_id"])
    check("crash_and_resume", led_c.snapshot(tid) == inc, f"crashed={crashed}", crashed)

    # 5. rollback restores the state before the run, and an older run cannot be rolled back under a newer one
    led_r = Ledger()
    r_base = ingest_tenant(led_r, cfg, base)
    state_after_base = led_r.snapshot(tid)
    r_sync = ingest_tenant(led_r, cfg, ctrl)
    refused = False
    try:
        rollback(led_r, r_base.run_id)
    except RollbackError:
        refused = True
    if r_sync.status == "committed":
        rollback(led_r, r_sync.run_id)
        check("rollback", led_r.snapshot(tid) == state_after_base and refused, f"older-run rollback refused={refused}", True)
    else:
        check("rollback", led_r.snapshot(tid) == state_after_base, "sync changed nothing", False)
    return out


def eval_ingest(tenants: list[Tenant], cfgs: dict[str, tuple[dict, list]]) -> dict:
    by: dict[str, list[int]] = {}
    failures = []
    for t in tenants:
        for name, ok, detail, exercised in ingest_checks(t, cfgs[t.tid][0]):
            row = by.setdefault(name, [0, 0, 0])
            row[1] += 1
            row[0] += int(ok)
            row[2] += int(exercised)
            if not ok:
                failures.append({"tenant": t.tid, "check": name, "detail": detail})
    return {"checks": sum(v[1] for v in by.values()), "passed": sum(v[0] for v in by.values()),
            "by_check": {k: {"passed": a, "of": b, "scenario_occurred": c} for k, (a, b, c) in by.items()}, "failures": failures}


# ------------------------------------------------------------------ main


def run_experiment(data_dir: Path, out_dir: Path, dev: bool = False, log=print) -> dict:
    t0 = time.time()
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not any(data_dir.glob("*/truth.json")):
        log(f"generating {data_dir} ...")
        generate_all(data_dir)
    tenants = load_tenants(data_dir)
    tuning, held = split_tenants(tenants)
    log(f"{len(tenants)} tenants: {len(tuning)} tuning, {len(held)} held out")

    # ---- 2. tuning (tuning tenants only)
    tuned, trials = tune_mapper(tuning)
    tsig = {t.tid: load_signals(t) for t in tuning}
    log(f"tuned mapper on tuning tenants: {tuned}")
    tuning_cfgs = {t.tid: predicted_config(t, tuned, tsig[t.tid]) for t in tuning}
    tau, drift_trials = tune_drift(tuning, {k: v[0] for k, v in tuning_cfgs.items()})
    log(f"tuned drift rename threshold: {tau}")
    trials += drift_trials
    with open(out_dir / "trials.jsonl", "w", encoding="utf-8") as fh:
        for tr in trials:
            fh.write(json.dumps(tr) + "\n")
    (out_dir / "tuned_params.json").write_text(json.dumps({"mapper": tuned.to_dict(), "drift_rename_threshold": tau}, indent=1), encoding="utf-8")
    baselines = baseline_systems(tuned, tuning, tsig)

    summary: dict = {"meta": {"version": __version__, "tenants": len(tenants), "tuning": [t.tid for t in tuning], "heldout": [t.tid for t in held],
                              "gen_seed": GEN_SEED, "split_seed": SPLIT_SEED, "python": platform.python_version(),
                              "lexicon_coverage": lexicon_coverage(tuning), "tuned_params": tuned.to_dict(), "drift_rename_threshold": tau,
                              "baseline_params": {k: v.to_dict() for k, v in baselines.items()}, "dev": dev}}

    def map_block(tenants_, signals, tables):
        systems = {"naive (exact header)": "naive", **{k: v for k, v in baselines.items()}, "tuned (all signals)": tuned}
        out = {}
        per = {k: _scores(s, tenants_, signals, tables) for k, s in systems.items()}
        for k, sc in per.items():
            out[k] = {"overall": aggregate(sc), "f1_ci": bootstrap_ci(sc), "by_difficulty": by_difficulty(tenants_, sc)}
        return out, per

    # ---- tuning split (reported for reference; this is the data the settings were chosen on)
    tables_t = {t.tid: [read_table(p) for p in t.files()] for t in tuning}
    summary["mapping_tuning_split"], _ = map_block(tuning, tsig, tables_t)
    if dev:
        (out_dir / "summary_dev.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        log(f"dev run done in {time.time() - t0:.0f}s (held-out tenants untouched)")
        return summary

    # ---- 3. held-out, scored once
    summary["meta"]["lexicon_coverage"] = lexicon_coverage(tenants)          # descriptive; reads headers of every tenant
    hsig = {t.tid: load_signals(t) for t in held}
    tables_h = {t.tid: [read_table(p) for p in t.files()] for t in held}
    summary["mapping"], per = map_block(held, hsig, tables_h)
    full_scores = per["tuned (all signals)"]
    abl = {}
    for name, params in ablations(tuned).items():
        sc = _scores(params, held, hsig, tables_h)
        a, f = aggregate(sc), aggregate(full_scores)
        abl[name] = {"f1": a["f1"], "delta_vs_full": round(a["f1"] - f["f1"], 4), "delta_ci": _paired_ci(sc, full_scores), "precision": a["precision"], "recall": a["recall"]}
    summary["ablation"] = abl
    summary["review"] = _review_stats(tuned, held, hsig)

    pred_cfgs = {t.tid: predicted_config(t, tuned, hsig[t.tid]) for t in held}
    orc_cfgs = {t.tid: oracle_config(t) for t in held}
    summary["normalization"] = {"predicted_mapping": eval_normalization(held, pred_cfgs), "oracle_mapping": eval_normalization(held, orc_cfgs)}
    summary["drift"] = {"predicted_mapping": eval_drift(held, pred_cfgs, tau), "oracle_mapping": eval_drift(held, orc_cfgs, tau),
                        "rename_threshold": tau}
    summary["ingest"] = eval_ingest(held, orc_cfgs)
    summary["meta"]["runtime_s"] = round(time.time() - t0, 1)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log(f"done in {time.time() - t0:.0f}s; wrote {out_dir / 'summary.json'}")
    return summary
