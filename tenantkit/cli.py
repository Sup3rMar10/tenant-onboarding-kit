"""Command line: demo, generate, map, ingest, drift, eval."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from .config import ConfigError, build_config, load_config, save_config, validate_config
from .drift import DriftParams, decision, detect_drift
from .generate import generate_all, load_tenants
from .ingest import InjectedFailure, Ledger, RollbackError, ingest_tenant, rollback
from .mapper import DEFAULT_PARAMS, Params, compute_signals, map_tenant
from .readers import read_table


def _files(args) -> list[Path]:
    out: list[Path] = []
    for a in args:
        p = Path(a)
        out += sorted(x for x in p.iterdir() if x.is_file() and x.suffix.lower() in (".csv", ".tsv", ".txt", ".json", ".ndjson", ".jsonl")) if p.is_dir() else [p]
    return out


def _params(path: str | None) -> Params:
    if not path:
        return DEFAULT_PARAMS
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Params(**data.get("mapper", data))


def print_mapping(mapping, tables) -> None:
    for t in tables:
        fm = mapping.files[t.name]
        meta = t.meta
        print(f"\n{t.name}  ({meta.get('kind')}, {meta.get('encoding')}"
              + (f", delimiter {meta['delimiter']!r}" if meta.get("delimiter") else "") + f", {len(t.rows)} rows)  ->  {fm.entity}")
        print(f"  {'source column':28s} {'target':16s} {'conf':>5s}  note")
        for c in fm.columns:
            if c.field:
                note = "REVIEW" if c.review else ""
                if c.decided_by not in ("score",):
                    note += f" ({c.decided_by})"
                print(f"  {c.source[:28]:28s} {c.field:16s} {c.score:5.2f}  {note}")
            else:
                best = f"best guess {c.alternatives[0][0].split('.')[1]} {c.alternatives[0][1]:.2f}" if c.alternatives and c.alternatives[0][1] >= 0.2 else ""
                print(f"  {c.source[:28]:28s} {'-':16s} {'':5s}  not mapped {best}")


def cmd_generate(a) -> int:
    plan = a.plan.split(",") if a.plan else None
    ts = generate_all(Path(a.out), n=len(plan) if plan else a.n, seed=a.seed, plan=plan)
    print(f"wrote {len(ts)} synthetic tenants to {a.out}")
    return 0


def cmd_map(a) -> int:
    files = _files(a.paths)
    tables = [read_table(p) for p in files]
    sig = compute_signals(tables)
    mapping = map_tenant(sig, _params(a.params))
    print_mapping(mapping, tables)
    cfg = build_config(a.tenant or files[0].parent.name, tables, mapping, sig)
    if a.out:
        save_config(cfg, a.out)
        probs = validate_config(cfg)
        print(f"\nwrote {a.out}" + (f"; to fix before ingest: {'; '.join(probs)}" if probs else "; passes validation"))
    return 0


def cmd_ingest(a) -> int:
    ledger = Ledger(a.db)
    if a.rollback is not None:
        try:
            print("rolled back:", rollback(ledger, a.rollback, force=a.force))
            return 0
        except RollbackError as e:
            print("refused:", e)
            return 2
    try:
        cfg = load_config(a.config)
    except ConfigError as e:
        print("config problems:")
        for p in e.problems:
            print(" -", p)
        return 2
    res = ingest_tenant(ledger, cfg, _files(a.paths), dry_run=a.dry_run, force=a.force, resume_run=a.resume)
    print(f"run {res.run_id}: {res.status}")
    for f in res.files:
        print("  ", {k: v for k, v in f.items()})
    for e in res.drift:
        print(f"   drift [{e.action}] {e.kind}: {e.detail}")
    for n in res.notes:
        print("  note:", n)
    if a.report and res.dq:
        print()
        print(res.dq.render())
    return 0 if res.status != "blocked" else 3


def cmd_drift(a) -> int:
    cfg = load_config(a.config)
    events = []
    for p in _files(a.paths):
        if p.name in cfg["files"]:
            events += detect_drift(cfg["files"][p.name], read_table(p), cfg["drift_policy"], DriftParams(cfg["thresholds"]["rename_confidence"]))
    for e in events:
        print(f"[{e.action:5s}] {e.file}: {e.kind}: {e.detail}")
    print(f"decision: {decision(events)}")
    return 0


def cmd_eval(a) -> int:
    from .experiment import run_experiment
    from .report import write_report

    out = Path(a.out)
    summary = run_experiment(Path(a.data), out, dev=a.dev)
    if not a.dev:
        write_report(out / "summary.json", out, Path(a.readme) if a.readme else None)
        print(f"wrote {out / 'REPORT.md'}")
    return 0


def cmd_report(a) -> int:
    from .report import write_report

    out = Path(a.summary).parent
    write_report(Path(a.summary), out, Path(a.readme) if a.readme else None)
    print(f"wrote charts and REPORT.md in {out}" + (f"; filled {a.readme}" if a.readme else ""))
    return 0


def cmd_demo(a) -> int:
    tmp = Path(tempfile.mkdtemp(prefix="tenantkit_demo_"))
    try:
        ts = generate_all(tmp / "data", n=3, plan=["easy", "medium", "hard"])
        t = ts[2]
        print(f"Three synthetic tenants generated. Following the hard one ({t.tid}): opaque headers, odd formats, junk rows.\n")
        files = t.files("base")
        tables = [read_table(p) for p in files]
        sig = compute_signals(tables)
        mapping = map_tenant(sig, DEFAULT_PARAMS)
        print("=== 1. mapping (header + value + relation evidence) ===")
        print_mapping(mapping, tables)
        cfg = build_config(t.tid, tables, mapping, sig)
        save_config(cfg, tmp / "config.json")
        print(f"\nconfig written to {tmp / 'config.json'} (a person reviews flagged columns and edits it)")
        ledger = Ledger(tmp / "ledger.db")
        print("\n=== 2. dry run, then load ===")
        dry = ingest_tenant(ledger, cfg, files, dry_run=True)
        print(f"dry run: {dry.status}, decision {dry.dq.decision}; ledger untouched: {len(ledger.snapshot(t.tid))} records")
        if dry.status == "blocked":
            print("blocked:", "; ".join(dry.notes))
            print("(a real customer would fix the flagged mapping in the config; the demo forces the load to continue)")
        first = ingest_tenant(ledger, cfg, files, force=True)
        print(f"load: {first.status}, run {first.run_id}, {len(ledger.snapshot(t.tid))} records")
        print("\n" + first.dq.render())
        again = ingest_tenant(ledger, cfg, files, force=True)
        print(f"\nsame export again: {again.status} (idempotent)")
        print("\n=== 3. the customer's next export: schema drift ===")
        events = []
        for p in t.files("drift"):
            if p.name in cfg["files"]:
                events += detect_drift(cfg["files"][p.name], read_table(p), cfg["drift_policy"])
        for e in events:
            print(f"  [{e.action:5s}] {e.file}: {e.kind}: {e.detail}")
        print(f"  decision: {decision(events)}")
        blocked = ingest_tenant(ledger, cfg, t.files("drift"))
        print(f"  ingest of that export: {blocked.status}")
        print("\n=== 4. a clean fuller re-export, then rollback ===")
        r = ingest_tenant(ledger, cfg, t.files("control"), force=True)
        print(f"  load: {r.status}; records now {len(ledger.snapshot(t.tid))}")
        print(f"  rollback run {r.run_id}: {rollback(ledger, r.run_id)}; records now {len(ledger.snapshot(t.tid))}")
        try:
            rollback(ledger, first.run_id)
        except RollbackError as e:
            print(f"  (rolling back an older run under a newer one is refused: {e})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tenantkit", description="Get a new customer's messy data into a product safely.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo", help="the whole flow on three synthetic tenants, offline, in seconds").set_defaults(fn=cmd_demo)
    g = sub.add_parser("generate", help="write synthetic tenants")
    g.add_argument("--out", default="data/tenants")
    g.add_argument("--n", type=int, default=40)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--plan", help="difficulties to generate, e.g. easy,medium,hard (default: the 40-tenant mix)")
    g.set_defaults(fn=cmd_generate)
    m = sub.add_parser("map", help="map a tenant's files to the canonical schema and write a reviewable config")
    m.add_argument("paths", nargs="+")
    m.add_argument("--out")
    m.add_argument("--tenant")
    m.add_argument("--params", help="JSON with mapper settings, e.g. results/tuned_params.json")
    m.set_defaults(fn=cmd_map)
    i = sub.add_parser("ingest", help="load files into the ledger using a config")
    i.add_argument("paths", nargs="*")
    i.add_argument("--config")
    i.add_argument("--db", default="ledger.db")
    i.add_argument("--dry-run", action="store_true")
    i.add_argument("--force", action="store_true", help="load despite a block decision")
    i.add_argument("--resume", type=int, help="resume a failed run")
    i.add_argument("--rollback", type=int, help="roll back a run")
    i.add_argument("--report", action="store_true", help="print the data-quality report")
    i.set_defaults(fn=cmd_ingest)
    d = sub.add_parser("drift", help="compare an export with the config's fingerprint")
    d.add_argument("paths", nargs="+")
    d.add_argument("--config", required=True)
    d.set_defaults(fn=cmd_drift)
    e = sub.add_parser("eval", help="the full evaluation on the generated tenants (generates them if missing)")
    e.add_argument("--data", default="data/tenants")
    e.add_argument("--out", default="results")
    e.add_argument("--dev", action="store_true", help="tune and score on the tuning tenants only; held-out tenants untouched")
    e.add_argument("--readme", help="README to fill between the RESULTS markers")
    e.set_defaults(fn=cmd_eval)
    r = sub.add_parser("report", help="rebuild charts, REPORT.md and the README results block from results/summary.json")
    r.add_argument("--summary", default="results/summary.json")
    r.add_argument("--readme")
    r.set_defaults(fn=cmd_report)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
