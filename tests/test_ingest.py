import copy
import json

import pytest

from tenantkit import ingest as ingest_mod
from tenantkit.config import build_config
from tenantkit.evaluate import oracle_mapping
from tenantkit.ingest import InjectedFailure, Ledger, RollbackError, ingest_tenant, rollback
from tenantkit.mapper import compute_signals
from tenantkit.readers import read_table


@pytest.fixture(scope="module")
def case(tenants):
    # an easy tenant whose control export actually changes things in every file
    for t in tenants:
        if t.difficulty != "easy":
            continue
        tables = [read_table(p) for p in t.files()]
        cfg = build_config(t.tid, tables, oracle_mapping(t.truth()), compute_signals(tables))
        led = Ledger()
        ingest_tenant(led, cfg, t.files("base"))
        r = ingest_tenant(led, cfg, t.files("control"))
        if r.status == "committed" and all(f.get("inserted") for f in r.files):
            return t, cfg
    pytest.skip("no suitable tenant")


def test_loading_the_same_export_twice_changes_nothing(case):
    t, cfg = case
    led = Ledger()
    first = ingest_tenant(led, cfg, t.files("base"))
    snap = led.snapshot(t.tid)
    again = ingest_tenant(led, cfg, t.files("base"))
    assert first.status == "committed" and again.status == "noop" and led.snapshot(t.tid) == snap
    assert all(f.get("skipped_unchanged") for f in again.files)


def test_a_fuller_export_inserts_new_rows_and_updates_only_changed_ones(case):
    t, cfg = case
    led = Ledger()
    ingest_tenant(led, cfg, t.files("base"))
    before = led.snapshot(t.tid)
    r = ingest_tenant(led, cfg, t.files("control"))
    after = led.snapshot(t.tid)
    ins = sum(f["inserted"] for f in r.files)
    upd = sum(f["updated"] for f in r.files)
    assert ins == len(set(after) - set(before)) > 0
    assert upd == sum(1 for k in before if after[k] != before[k])
    changes = led.db.execute("SELECT COUNT(*) FROM changes WHERE run_id=?", (r.run_id,)).fetchone()[0]
    assert changes == ins + upd                                      # every write is journalled


def test_dry_run_writes_nothing_and_predicts_the_real_run(case):
    t, cfg = case
    led = Ledger()
    ingest_tenant(led, cfg, t.files("base"))
    snap, runs = led.snapshot(t.tid), led.runs(t.tid)
    dry = ingest_tenant(led, cfg, t.files("control"), dry_run=True)
    assert dry.status == "dry_run" and led.snapshot(t.tid) == snap and led.runs(t.tid) == runs
    assert led.db.execute("SELECT COUNT(*) FROM changes").fetchone()[0] == led.db.execute("SELECT COUNT(*) FROM changes WHERE run_id=1").fetchone()[0]
    real = ingest_tenant(led, cfg, t.files("control"))
    pick = lambda r: sorted((f["file"], f["inserted"], f["updated"]) for f in r.files)
    assert pick(dry) == pick(real)


def test_a_crash_between_files_is_checkpointed_and_resumable(case):
    t, cfg = case
    clean = Ledger()
    ingest_tenant(clean, cfg, t.files("base"))
    ingest_tenant(clean, cfg, t.files("control"))
    led = Ledger()
    ingest_tenant(led, cfg, t.files("base"))
    with pytest.raises(InjectedFailure):
        ingest_tenant(led, cfg, t.files("control"), fail_after=1)
    failed = [r for r in led.runs(t.tid) if r["status"] == "failed"]
    assert len(failed) == 1
    done = led.db.execute("SELECT filename FROM run_files WHERE run_id=? AND status='committed'", (failed[0]["run_id"],)).fetchall()
    assert len(done) == 1                                            # the first file is durable, the rest not started
    assert led.snapshot(t.tid) != clean.snapshot(t.tid)
    res = ingest_tenant(led, cfg, t.files("control"), resume_run=failed[0]["run_id"])
    assert any(f.get("skipped_checkpoint") for f in res.files) and res.status == "committed"
    assert led.snapshot(t.tid) == clean.snapshot(t.tid)
    with pytest.raises(ValueError):
        ingest_tenant(led, cfg, t.files("control"), resume_run=failed[0]["run_id"])          # no longer failed


def test_a_file_is_all_or_nothing(case, monkeypatch):
    t, cfg = case
    real = ingest_mod._load_file
    calls = {"n": 0}

    def flaky(db, tenant, run_id, nf):
        calls["n"] += 1
        if calls["n"] == 2:
            real(db, tenant, run_id, type(nf)(nf.filename, nf.entity, nf.records[: len(nf.records) // 2], nf.issues, nf.quarantined, nf.stats))
            raise RuntimeError("disk full")
        return real(db, tenant, run_id, nf)

    monkeypatch.setattr(ingest_mod, "_load_file", flaky)
    led = Ledger()
    with pytest.raises(RuntimeError):
        ingest_tenant(led, cfg, t.files("base"))
    entities = {e for (e, _k) in led.snapshot(t.tid)}
    assert entities == {"customer"}                                  # file 1 committed, file 2's half-written rows rolled back
    assert led.runs(t.tid)[0]["status"] == "failed"


def test_rollback_restores_previous_content_and_is_last_in_first_out(case):
    t, cfg = case
    led = Ledger()
    r1 = ingest_tenant(led, cfg, t.files("base"))
    s1 = led.snapshot(t.tid)
    r2 = ingest_tenant(led, cfg, t.files("control"))
    assert led.snapshot(t.tid) != s1
    with pytest.raises(RollbackError, match="roll that back first"):
        rollback(led, r1.run_id)
    out = rollback(led, r2.run_id)
    assert out["deleted"] > 0 and led.snapshot(t.tid) == s1
    assert led.run(r2.run_id)["status"] == "rolled_back"
    with pytest.raises(RollbackError):
        rollback(led, r2.run_id)                                     # already rolled back
    again = ingest_tenant(led, cfg, t.files("control"))              # its content counts as not loaded any more
    assert again.status == "committed" and led.snapshot(t.tid) != s1
    rollback(led, again.run_id)
    rollback(led, r1.run_id)
    assert led.snapshot(t.tid) == {}


def test_force_can_roll_back_under_a_newer_run(case):
    t, cfg = case
    led = Ledger()
    r1 = ingest_tenant(led, cfg, t.files("base"))
    ingest_tenant(led, cfg, t.files("control"))
    rollback(led, r1.run_id, force=True)
    assert led.run(r1.run_id)["status"] == "rolled_back"


def test_blocking_drift_stops_the_run_before_anything_is_written(tenants):
    for t in tenants:
        tables = [read_table(p) for p in t.files()]
        cfg = build_config(t.tid, tables, oracle_mapping(t.truth()), compute_signals(tables))
        if not any(e["kind"] in ("drop_required", "retype_cents", "id_shape") for e in t.truth("drift")["events"]):
            continue
        led = Ledger()
        ingest_tenant(led, cfg, t.files("base"))
        snap = led.snapshot(t.tid)
        res = ingest_tenant(led, cfg, t.files("drift"))
        assert res.status == "blocked" and led.snapshot(t.tid) == snap
        assert any("drift" in n for n in res.notes) and led.runs(t.tid)[-1]["status"] == "blocked"
        forced = ingest_tenant(led, cfg, t.files("drift"), force=True)
        assert forced.status in ("committed", "noop")
        return
    pytest.skip("no blocking drift in fixture")


def test_data_quality_can_block_and_quarantine_is_recorded(case):
    t, cfg = case
    bad = copy.deepcopy(cfg)
    fn = next(n for n in bad["files"] if n.startswith("invoices"))
    for c in bad["files"][fn]["columns"]:
        if c["target"] == "amount":
            c["transform"] = {"type": "date", "format": "%Y-%m-%d"}
    led = Ledger()
    res = ingest_tenant(led, cfg=bad, paths=t.files("base"))
    assert res.status == "blocked" and any("data quality" in n for n in res.notes) and led.snapshot(t.tid) == {}


def test_quarantine_table_has_reasons_not_values(tenants):
    for t in [t for t in tenants if t.difficulty == "hard" and t.truth()["expected_quarantine"]["invoice"]]:
        tables = [read_table(p) for p in t.files()]
        cfg = build_config(t.tid, tables, oracle_mapping(t.truth()), compute_signals(tables))
        led = Ledger()
        ingest_tenant(led, cfg, t.files("base"), force=True)
        rows = led.db.execute("SELECT filename, row_no, reason FROM quarantine").fetchall()
        assert rows and all(":" in r[2] and "TBD" not in r[2] for r in rows)
        return
    pytest.skip("no tenant with junk amounts in the fixture")


def test_runs_carry_an_audit_trail(case):
    t, cfg = case
    led = Ledger()
    ingest_tenant(led, cfg, t.files("base"))
    run = led.runs(t.tid)[0]
    stats = json.loads(run["stats"])
    assert run["status"] == "committed" and run["started"] and run["finished"] and {f["file"] for f in stats["files"]} == {p.name for p in t.files("base")}
