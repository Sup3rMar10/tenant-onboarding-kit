"""Idempotent, incremental ingest into a SQLite ledger, with dry runs, checkpoints and rollback.

Guarantees, each one covered by a test:

  idempotent    loading the same export twice changes nothing. A file whose content hash matches
                the last committed load of that file is skipped; a changed file is applied as an
                upsert keyed by (tenant, entity, key), and only rows whose content changed are written.
  incremental   a later, fuller export updates changed rows and inserts new ones; untouched rows keep
                their history.
  atomic files  each file loads in one transaction that also records its checkpoint, so a crash leaves
                every file either fully loaded or not at all.
  resumable     a failed run can be resumed: files already checkpointed are skipped.
  dry run       does everything, writes nothing (the whole run is one transaction that is rolled back).
  reversible    every insert and update is journalled with the row's previous content, so a run can be
                rolled back. Rollback is last-in-first-out per tenant, so one run cannot silently
                resurrect rows a later run has changed (override with force).
  gated         schema drift with a `block` policy, or a data-quality report whose decision is `block`,
                stops the run before anything is written, unless forced.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .drift import DriftEvent, DriftParams, apply_heals, decision, detect_drift
from .normalize import NormalizedFile, normalize_table, record_hash
from .readers import RawTable, read_table
from .validate import DQReport, validate

ENTITY_ORDER = ["customer", "invoice", "payment"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(run_id INTEGER PRIMARY KEY AUTOINCREMENT, tenant TEXT, started TEXT, finished TEXT,
                                status TEXT, note TEXT, stats TEXT);
CREATE TABLE IF NOT EXISTS run_files(run_id INTEGER, filename TEXT, sha256 TEXT, entity TEXT, rows_in INTEGER, inserted INTEGER,
                                     updated INTEGER, unchanged INTEGER, quarantined INTEGER, status TEXT, PRIMARY KEY(run_id, filename));
CREATE TABLE IF NOT EXISTS records(tenant TEXT, entity TEXT, key TEXT, data TEXT, row_hash TEXT, first_run INTEGER, last_run INTEGER,
                                   PRIMARY KEY(tenant, entity, key));
CREATE TABLE IF NOT EXISTS changes(change_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER, tenant TEXT, entity TEXT, key TEXT,
                                   op TEXT, before TEXT);
CREATE TABLE IF NOT EXISTS quarantine(run_id INTEGER, tenant TEXT, filename TEXT, row_no INTEGER, reason TEXT);
"""


class InjectedFailure(RuntimeError):
    """Raised on purpose by tests and the evaluation to simulate a crash between files."""


class RollbackError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Ledger:
    def __init__(self, path: str | Path = ":memory:"):
        self.db = sqlite3.connect(str(path), isolation_level=None)       # explicit transactions
        self.db.executescript(SCHEMA)

    def snapshot(self, tenant: str) -> dict[tuple[str, str], str]:
        rows = self.db.execute("SELECT entity, key, row_hash FROM records WHERE tenant=?", (tenant,)).fetchall()
        return {(e, k): h for e, k, h in rows}

    def keys(self, tenant: str) -> dict[str, set]:
        out: dict[str, set] = {e: set() for e in ENTITY_ORDER}
        for e, k in self.db.execute("SELECT entity, key FROM records WHERE tenant=?", (tenant,)):
            out[e].add(k)
        return out

    def run(self, run_id: int) -> dict | None:
        r = self.db.execute("SELECT run_id, tenant, started, finished, status, note, stats FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return dict(zip(("run_id", "tenant", "started", "finished", "status", "note", "stats"), r)) if r else None

    def runs(self, tenant: str) -> list[dict]:
        ids = [r[0] for r in self.db.execute("SELECT run_id FROM runs WHERE tenant=? ORDER BY run_id", (tenant,))]
        return [self.run(i) for i in ids]

    def last_loaded_hash(self, tenant: str, filename: str) -> str | None:
        r = self.db.execute(
            "SELECT rf.sha256 FROM run_files rf JOIN runs r ON r.run_id=rf.run_id WHERE r.tenant=? AND rf.filename=? "
            "AND rf.status='committed' AND r.status IN ('committed','noop') ORDER BY rf.run_id DESC LIMIT 1", (tenant, filename)).fetchone()
        return r[0] if r else None


@dataclass
class RunResult:
    run_id: int | None
    status: str                      # committed | noop | dry_run | blocked | failed
    files: list[dict] = field(default_factory=list)
    drift: list[DriftEvent] = field(default_factory=list)
    dq: DQReport | None = None
    healed: bool = False
    notes: list[str] = field(default_factory=list)


def _load_file(db: sqlite3.Connection, tenant: str, run_id: int, nf: NormalizedFile) -> dict:
    ins = upd = same = 0
    for rec in nf.records:
        key = rec[f"{nf.entity}_id"]
        h = record_hash(rec)
        data = json.dumps({k: v for k, v in rec.items() if not k.startswith("_")}, sort_keys=True, ensure_ascii=False)
        cur = db.execute("SELECT data, row_hash, first_run, last_run FROM records WHERE tenant=? AND entity=? AND key=?",
                         (tenant, nf.entity, key)).fetchone()
        if cur is None:
            db.execute("INSERT INTO records VALUES (?,?,?,?,?,?,?)", (tenant, nf.entity, key, data, h, run_id, run_id))
            db.execute("INSERT INTO changes(run_id, tenant, entity, key, op, before) VALUES (?,?,?,?,?,NULL)", (run_id, tenant, nf.entity, key, "insert"))
            ins += 1
        elif cur[1] != h:
            db.execute("INSERT INTO changes(run_id, tenant, entity, key, op, before) VALUES (?,?,?,?,?,?)",
                       (run_id, tenant, nf.entity, key, "update", json.dumps(list(cur))))
            db.execute("UPDATE records SET data=?, row_hash=?, last_run=? WHERE tenant=? AND entity=? AND key=?", (data, h, run_id, tenant, nf.entity, key))
            upd += 1
        else:
            same += 1
    for row_no, reason in nf.quarantined:
        db.execute("INSERT INTO quarantine VALUES (?,?,?,?,?)", (run_id, tenant, nf.filename, row_no, reason))
    return {"inserted": ins, "updated": upd, "unchanged": same, "quarantined": len(nf.quarantined)}


def _prepare(ledger: Ledger, cfg: dict, paths: list[Path]):
    tenant = cfg["tenant_id"]
    tables: dict[str, RawTable] = {}
    notes: list[str] = []
    for p in paths:
        if p.name not in cfg["files"]:
            notes.append(f"{p.name}: not in the config, skipped")
            continue
        tables[p.name] = read_table(p)
    for fname in cfg["files"]:
        if fname not in tables and fname not in [p.name for p in paths]:
            notes.append(f"{fname}: in the config but not in this export")
    params = DriftParams(cfg.get("thresholds", {}).get("rename_confidence", DriftParams().rename_threshold))
    events: list[DriftEvent] = []
    eff_files: dict[str, dict] = {}
    for fname, t in tables.items():
        ev = detect_drift(cfg["files"][fname], t, cfg.get("drift_policy"), params)
        events += ev
        eff_files[fname] = apply_heals(cfg["files"][fname], ev)
    normalized = {fname: normalize_table(t, eff_files[fname]) for fname, t in tables.items()}
    report = validate(tenant, normalized, ledger.keys(tenant), cfg.get("thresholds"))
    return tables, events, normalized, report, notes


def ingest_tenant(ledger: Ledger, cfg: dict, paths: list[Path | str], *, dry_run: bool = False, fail_after: int | None = None,
                  resume_run: int | None = None, force: bool = False) -> RunResult:
    paths = [Path(p) for p in paths]
    tenant = cfg["tenant_id"]
    db = ledger.db
    tables, events, normalized, report, notes = _prepare(ledger, cfg, paths)
    result = RunResult(None, "committed", drift=events, dq=report, notes=notes, healed=any(e.action == "heal" for e in events))
    blockers = []
    if decision(events) == "block":
        blockers += [f"drift: {e.detail}" for e in events if e.action == "block"]
    if report.decision == "block":
        blockers += [f"data quality: {f.message}" for f in report.findings if f.level == "block"]
    if blockers and not force:
        result.status, result.notes = "blocked", notes + blockers
        if not dry_run:
            cur = db.execute("INSERT INTO runs(tenant, started, finished, status, note, stats) VALUES (?,?,?,?,?,?)",
                             (tenant, _now(), _now(), "blocked", "; ".join(blockers)[:1000], "{}"))
            result.run_id = cur.lastrowid
        return result

    ordered = sorted(normalized.values(), key=lambda nf: ENTITY_ORDER.index(nf.entity))
    hashes = {fname: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths for fname in [p.name] if fname in normalized}

    if dry_run:
        db.execute("BEGIN")
        try:
            run_id = db.execute("INSERT INTO runs(tenant, started, status, note, stats) VALUES (?,?,?,?,?)",
                                (tenant, _now(), "dry_run", "", "{}")).lastrowid
            for nf in ordered:
                unchanged_file = ledger.last_loaded_hash(tenant, nf.filename) == hashes[nf.filename]
                stats = {"inserted": 0, "updated": 0, "unchanged": nf.stats["loaded"], "quarantined": nf.stats["quarantined"]} \
                    if unchanged_file else _load_file(db, tenant, run_id, nf)
                result.files.append({"file": nf.filename, "entity": nf.entity, "skipped_unchanged": unchanged_file, **stats})
        finally:
            db.execute("ROLLBACK")
        result.status = "dry_run"
        return result

    if resume_run is not None:
        run = ledger.run(resume_run)
        if run is None or run["tenant"] != tenant or run["status"] != "failed":
            raise ValueError(f"run {resume_run} is not a failed run of tenant {tenant}")
        run_id = resume_run
        db.execute("UPDATE runs SET status='running' WHERE run_id=?", (run_id,))
    else:
        run_id = db.execute("INSERT INTO runs(tenant, started, status, note, stats) VALUES (?,?,?,?,?)",
                            (tenant, _now(), "running", "", "{}")).lastrowid
    result.run_id = run_id
    done = {r[0] for r in db.execute("SELECT filename FROM run_files WHERE run_id=? AND status='committed'", (run_id,))}
    committed_now = 0
    try:
        for nf in ordered:
            if nf.filename in done:
                result.files.append({"file": nf.filename, "entity": nf.entity, "skipped_checkpoint": True})
                continue
            sha = hashes[nf.filename]
            if ledger.last_loaded_hash(tenant, nf.filename) == sha:
                db.execute("INSERT INTO run_files VALUES (?,?,?,?,?,?,?,?,?,?)",
                           (run_id, nf.filename, sha, nf.entity, nf.stats["rows_in"], 0, 0, nf.stats["loaded"], nf.stats["quarantined"], "unchanged"))
                result.files.append({"file": nf.filename, "entity": nf.entity, "skipped_unchanged": True})
                continue
            db.execute("BEGIN")
            try:
                st = _load_file(db, tenant, run_id, nf)
                db.execute("INSERT INTO run_files VALUES (?,?,?,?,?,?,?,?,?,?)",
                           (run_id, nf.filename, sha, nf.entity, nf.stats["rows_in"], st["inserted"], st["updated"], st["unchanged"], st["quarantined"], "committed"))
                db.execute("COMMIT")
            except Exception:
                db.execute("ROLLBACK")
                raise
            committed_now += 1
            result.files.append({"file": nf.filename, "entity": nf.entity, **st})
            if fail_after is not None and committed_now >= fail_after and nf is not ordered[-1]:
                raise InjectedFailure(f"injected failure after {committed_now} file(s)")
    except Exception as exc:
        db.execute("UPDATE runs SET status='failed', finished=?, note=? WHERE run_id=?", (_now(), f"{type(exc).__name__}: {exc}"[:500], run_id))
        result.status = "failed"
        raise
    changed = any(f.get("inserted") or f.get("updated") for f in result.files)
    result.status = "committed" if changed or any(f.get("skipped_checkpoint") for f in result.files) else "noop"
    db.execute("UPDATE runs SET status=?, finished=?, stats=? WHERE run_id=?",
               (result.status, _now(), json.dumps({"files": result.files, "drift": [e.kind for e in events]}), run_id))
    return result


def rollback(ledger: Ledger, run_id: int, force: bool = False) -> dict:
    db = ledger.db
    run = ledger.run(run_id)
    if run is None:
        raise RollbackError(f"no such run: {run_id}")
    if run["status"] not in ("committed", "failed", "noop"):
        raise RollbackError(f"run {run_id} is {run['status']}, nothing to roll back")
    later = db.execute("SELECT r.run_id FROM runs r WHERE r.tenant=? AND r.run_id>? AND r.status IN ('committed','failed') "
                       "AND EXISTS (SELECT 1 FROM changes c WHERE c.run_id=r.run_id) ORDER BY r.run_id LIMIT 1", (run["tenant"], run_id)).fetchone()
    if later and not force:
        raise RollbackError(f"run {later[0]} came after run {run_id} and changed this tenant; roll that back first (or force)")
    reverted = {"deleted": 0, "restored": 0}
    db.execute("BEGIN")
    try:
        for change_id, entity, key, op, before in db.execute(
                "SELECT change_id, entity, key, op, before FROM changes WHERE run_id=? ORDER BY change_id DESC", (run_id,)).fetchall():
            if op == "insert":
                db.execute("DELETE FROM records WHERE tenant=? AND entity=? AND key=?", (run["tenant"], entity, key))
                reverted["deleted"] += 1
            else:
                data, h, first, last = json.loads(before)
                db.execute("UPDATE records SET data=?, row_hash=?, first_run=?, last_run=? WHERE tenant=? AND entity=? AND key=?",
                           (data, h, first, last, run["tenant"], entity, key))
                reverted["restored"] += 1
        db.execute("UPDATE runs SET status='rolled_back', finished=? WHERE run_id=?", (_now(), run_id))
        db.execute("UPDATE run_files SET status='rolled_back' WHERE run_id=?", (run_id,))
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        raise
    return reverted
