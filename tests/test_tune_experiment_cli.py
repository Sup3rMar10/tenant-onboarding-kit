import json
import shutil
from pathlib import Path

import pytest

from tenantkit import cli
from tenantkit.evaluate import SPLIT_SEED, split_tenants
from tenantkit.generate import DIFFICULTY_PLAN, load_tenants
from tenantkit.mapper import DEFAULT_PARAMS
from tenantkit.tune import tune_mapper

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ the split

def test_split_is_disjoint_stable_and_stratified(tenants):
    tuning, held = split_tenants(tenants)
    assert {t.tid for t in tuning}.isdisjoint({t.tid for t in held}) and len(tuning) + len(held) == len(tenants)
    assert split_tenants(list(reversed(tenants))) == (tuning, held)            # independent of input order
    for diff in ("easy", "medium", "hard"):
        assert any(t.difficulty == diff for t in tuning) and any(t.difficulty == diff for t in held)
    assert split_tenants(tenants, seed=SPLIT_SEED + 1) != (tuning, held)


def test_full_size_split_is_balanced():
    from types import SimpleNamespace

    fake = [SimpleNamespace(tid=f"t{i:02d}", difficulty=d) for i, d in enumerate(DIFFICULTY_PLAN)]
    tuning, held = split_tenants(fake)
    assert (len(tuning), len(held)) == (20, 20)


# ------------------------------------------------------------------ no leakage from held-out truth

def _poison(truth_path: Path) -> None:
    t = json.loads(truth_path.read_text(encoding="utf-8"))
    for meta in t["files"].values():
        cols = list(meta["columns"])
        shifted = [meta["columns"][c] for c in cols[1:]] + [meta["columns"][cols[0]]]     # every label now wrong
        meta["columns"] = dict(zip(cols, shifted))
    truth_path.write_text(json.dumps(t), encoding="utf-8")


def test_held_out_truth_cannot_change_what_tuning_chooses(small_data, tmp_path):
    poisoned = tmp_path / "poisoned"
    shutil.copytree(small_data, poisoned)
    _, held = split_tenants(load_tenants(poisoned))
    for t in held:
        for f in t.root.glob("truth*.json"):
            _poison(f)
    p1, trials1 = tune_mapper(split_tenants(load_tenants(small_data))[0])
    p2, trials2 = tune_mapper(split_tenants(load_tenants(poisoned))[0])
    assert p1 == p2 and [t["f1"] for t in trials1] == [t["f1"] for t in trials2]


def test_the_poison_would_matter_if_it_reached_tuning(small_data, tmp_path):
    """Control for the test above: poisoning the tuning truth does change the search."""
    poisoned = tmp_path / "p2"
    shutil.copytree(small_data, poisoned)
    tuning, _ = split_tenants(load_tenants(poisoned))
    for t in tuning:
        _poison(t.root / "truth.json")
    clean_trials = tune_mapper(split_tenants(load_tenants(small_data))[0])[1]
    bad_trials = tune_mapper(split_tenants(load_tenants(poisoned))[0])[1]
    assert [t["f1"] for t in clean_trials] != [t["f1"] for t in bad_trials]


def test_trials_are_logged_and_tuning_is_a_coordinate_search(small_data, tmp_path):
    tuning, _ = split_tenants(load_tenants(small_data))
    path = tmp_path / "trials.jsonl"
    params, trials = tune_mapper(tuning, trials_path=path)
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    assert len(lines) == len(trials) > 40 and lines[0]["stage"] == "start"
    assert {l["stage"].split(":")[1] for l in lines[1:]} >= {"w_header", "w_rel", "threshold", "date_rule"}
    assert max(l["f1"] for l in lines) >= lines[0]["f1"]


def test_shipped_defaults_are_the_tuned_values():
    path = ROOT / "results" / "tuned_params.json"
    if not path.exists():
        pytest.skip("results/ not generated yet")
    tuned = json.loads(path.read_text(encoding="utf-8"))
    assert tuned["mapper"] == DEFAULT_PARAMS.to_dict()


# ------------------------------------------------------------------ the whole experiment, small

@pytest.fixture(scope="module")
def small_run(small_data, tmp_path_factory):
    from tenantkit.experiment import run_experiment

    out = tmp_path_factory.mktemp("results")
    return run_experiment(small_data, out, log=lambda *_: None), out


def test_experiment_runs_end_to_end_and_reports_everything(small_run):
    summary, out = small_run
    for key in ("mapping", "mapping_tuning_split", "ablation", "review", "normalization", "drift", "ingest", "meta"):
        assert key in summary
    assert (out / "summary.json").exists() and (out / "trials.jsonl").exists() and (out / "tuned_params.json").exists()
    m = summary["mapping"]
    assert m["tuned (all signals)"]["overall"]["f1"] > m["naive (exact header)"]["overall"]["f1"]
    assert summary["ingest"]["passed"] == summary["ingest"]["checks"] and not summary["ingest"]["failures"]
    assert summary["drift"]["oracle_mapping"]["control_false_alarms"] == 0
    assert set(summary["meta"]["heldout"]).isdisjoint(summary["meta"]["tuning"])


def test_report_writes_charts_tables_and_fills_the_readme(small_run, tmp_path):
    from tenantkit.report import MARK_END, MARK_START, write_report

    summary, out = small_run
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{MARK_START}\nPLACEHOLDER-XYZ\n{MARK_END}\noutro\n", encoding="utf-8")
    write_report(out / "summary.json", out, readme)
    text = readme.read_text(encoding="utf-8")
    assert "PLACEHOLDER-XYZ" not in text and text.startswith("intro") and text.rstrip().endswith("outro") and "Column mapping on held-out tenants" in text
    for png in ("mapping_by_difficulty.png", "ablation.png", "drift.png"):
        assert (out / png).stat().st_size > 5000


def test_dev_mode_never_touches_held_out_truth(small_data, tmp_path):
    from tenantkit.experiment import run_experiment

    data = tmp_path / "d"
    shutil.copytree(small_data, data)
    _, held = split_tenants(load_tenants(data))
    for t in held:
        for f in t.root.glob("truth*.json"):
            f.write_text("not json at all", encoding="utf-8")        # opening any of these would raise
    out = tmp_path / "out"
    summary = run_experiment(data, out, dev=True, log=lambda *_: None)
    assert "mapping" not in summary and (out / "summary_dev.json").exists()
    clean = run_experiment(small_data, tmp_path / "out2", dev=True, log=lambda *_: None)
    assert summary["meta"]["tuned_params"] == clean["meta"]["tuned_params"]


# ------------------------------------------------------------------ CLI

def test_demo_runs_offline(capsys):
    assert cli.main(["demo"]) == 0
    out = capsys.readouterr().out
    for phrase in ("1. mapping", "dry run", "idempotent", "schema drift", "rollback run"):
        assert phrase in out


def test_map_ingest_drift_rollback_commands(easy_tenant, tmp_path, capsys):
    cfg, db = tmp_path / "cfg.json", tmp_path / "ledger.db"
    base = str(easy_tenant.data_dir("base"))
    assert cli.main(["map", base, "--out", str(cfg), "--tenant", "t"]) == 0
    assert "passes validation" in capsys.readouterr().out
    assert cli.main(["ingest", base, "--config", str(cfg), "--db", str(db), "--report"]) == 0
    out = capsys.readouterr().out
    assert "committed" in out and "Data quality report" in out
    assert cli.main(["drift", str(easy_tenant.data_dir("control")), "--config", str(cfg)]) == 0
    assert "decision: clean" in capsys.readouterr().out
    assert cli.main(["ingest", str(easy_tenant.data_dir("control")), "--config", str(cfg), "--db", str(db)]) == 0
    capsys.readouterr()
    assert cli.main(["ingest", "--db", str(db), "--rollback", "1"]) == 2                  # run 1 is older than run 2
    assert "refused" in capsys.readouterr().out
    assert cli.main(["ingest", "--db", str(db), "--rollback", "2"]) == 0


def test_ingest_refuses_a_config_a_human_has_not_finished(easy_tenant, tmp_path, capsys):
    cfg = tmp_path / "cfg.json"
    cli.main(["map", str(easy_tenant.data_dir("base")), "--out", str(cfg)])
    data = json.loads(cfg.read_text())
    next(c for c in next(iter(data["files"].values()))["columns"] if c["target"] in ("name", "customer_id"))["target"] = None
    cfg.write_text(json.dumps(data))
    capsys.readouterr()
    assert cli.main(["ingest", str(easy_tenant.data_dir("base")), "--config", str(cfg), "--db", str(tmp_path / "l.db")]) == 2
    assert "config problems" in capsys.readouterr().out
