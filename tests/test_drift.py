import copy
import random
from datetime import date, timedelta

import pytest

from tenantkit.config import build_config
from tenantkit.drift import DriftParams, apply_heals, decision, detect_drift, similarity
from tenantkit.generate import generate_all
from tenantkit.mapper import Params, compute_signals, map_tenant
from tenantkit.normalize import normalize_table
from tenantkit.readers import RawTable, read_table
from tenantkit.tune import detect_for_tenant, drift_truth_keys

COLS = ["invoice_id", "customer_id", "invoice_date", "due_date", "currency", "amount", "status", "notes"]


def _rows(n=60, seed=1, date_fmt="%Y-%m-%d", amount=lambda v: f"{v:.2f}"):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        d = date(2025, 1, 1) + timedelta(days=rng.randint(0, 300))
        out.append([f"INV{40000 + i}", f"C{rng.randint(1, 20):04d}", d.strftime(date_fmt), (d + timedelta(days=30)).strftime(date_fmt), "EUR",
                    amount(rng.randint(1000, 900000) / 100), rng.choice(["open", "paid", "overdue"]), rng.choice(["ok", "call", "vip"])])
    return out


def _table(cols=COLS, rows=None, name="invoices.csv"):
    return RawTable(name, list(cols), rows if rows is not None else _rows(), {"kind": "delimited", "encoding": "utf-8", "delimiter": ","})


@pytest.fixture(scope="module")
def base_cfg():
    base = _table()
    cust = RawTable("customers.csv", ["customer_id", "name"], [[f"C{i:04d}", f"Acme {i} Inc"] for i in range(1, 21)],
                    {"kind": "delimited", "encoding": "utf-8", "delimiter": ","})
    sig = compute_signals([cust, base])
    cfg = build_config("t", [cust, base], map_tenant(sig, Params()), sig)
    assert {c["source"]: c["target"] for c in cfg["files"]["invoices.csv"]["columns"]}["amount"] == "amount"
    return cfg["files"]["invoices.csv"]


def _modify(rows, cols, fn):
    return [fn(dict(zip(cols, r))) for r in rows]


def _events(fcfg, table, **kw):
    return detect_drift(fcfg, table, **kw)


def test_new_rows_and_changed_values_are_not_drift(base_cfg):
    rows = _rows(n=80, seed=7)                    # more rows, different values, same shape
    for r in rows[:10]:
        r[6] = "paid"
    assert _events(base_cfg, _table(rows=rows)) == []


def test_rename_is_recognised_by_content_and_healed(base_cfg):
    cols = [("Belegdatum" if c == "invoice_date" else c) for c in COLS]
    t = _table(cols, _rows())
    ev = _events(base_cfg, t)
    assert [(e.kind, e.column, e.new_column, e.action) for e in ev] == [("renamed", "invoice_date", "Belegdatum", "heal")]
    healed = apply_heals(base_cfg, ev)
    out = normalize_table(t, healed)
    assert out.stats["quarantined"] == 0 and out.records[0]["invoice_date"]
    healed_col = next(c for c in healed["columns"] if c.get("healed"))
    assert healed_col["source"] == "Belegdatum" and healed_col["needs_review"]
    assert "Belegdatum" in healed["baseline"] and "invoice_date" not in healed["baseline"]
    assert base_cfg["columns"] != healed["columns"]                          # a copy: the original is untouched


def test_dropped_columns_are_required_or_optional(base_cfg):
    t = _table([c for c in COLS if c != "amount"], [[v for c, v in zip(COLS, r) if c != "amount"] for r in _rows()])
    ev = _events(base_cfg, t)
    assert [(e.kind, e.action, e.target) for e in ev] == [("dropped_required", "block", "amount")] and decision(ev) == "block"
    t2 = _table([c for c in COLS if c != "notes"], [[v for c, v in zip(COLS, r) if c != "notes"] for r in _rows()])
    ev2 = _events(base_cfg, t2)
    assert [(e.kind, e.action) for e in ev2] == [("dropped_optional", "alert")] and decision(ev2) == "alert"


def test_added_column_is_an_alert_and_may_come_with_a_suggestion(base_cfg):
    cols = COLS + ["payment_terms"]
    t = _table(cols, [r + ["Net 30"] for r in _rows()])
    ev = _events(base_cfg, t)
    assert [(e.kind, e.action, e.column) for e in ev] == [("added", "alert", "payment_terms")]


def test_retyped_date_format(base_cfg):
    t = _table(rows=_rows(date_fmt="%d.%m.%Y"))
    kinds = {(e.kind, e.column) for e in _events(base_cfg, t)}
    assert kinds == {("retyped_format", "invoice_date"), ("retyped_format", "due_date")}
    healed = apply_heals(base_cfg, _events(base_cfg, t))
    assert normalize_table(t, healed).stats["quarantined"] == 0


def test_retyped_decimal_separator(base_cfg):
    t = _table(rows=_rows(amount=lambda v: f"{v:.2f}".replace(".", ",")))
    ev = _events(base_cfg, t)
    assert [(e.kind, e.column) for e in ev] == [("retyped_format", "amount")] and ev[0].action == "heal"
    out = normalize_table(t, apply_heals(base_cfg, ev))
    assert out.stats["quarantined"] == 0


def test_silent_change_to_cents_is_blocked(base_cfg):
    t = _table(rows=_rows(amount=lambda v: str(round(v * 100))))
    ev = _events(base_cfg, t)
    assert [(e.kind, e.action) for e in ev] == [("retyped_incompatible", "block")]
    assert "money" in ev[0].detail


def test_id_shape_or_length_change_is_blocked(base_cfg):
    rows = _rows()
    for r in rows:
        r[0] = r[0].replace("INV", "")           # INV40001 -> 40001
    ev = _events(base_cfg, _table(rows=rows))
    assert [(e.kind, e.column, e.action) for e in ev] == [("retyped_incompatible", "invoice_id", "block")]


def test_the_policy_table_is_the_customers_to_change(base_cfg):
    t = _table([c for c in COLS if c != "amount"], [[v for c, v in zip(COLS, r) if c != "amount"] for r in _rows()])
    ev = _events(base_cfg, t, policy={"dropped_required": "alert"})
    assert ev[0].action == "alert" and decision(ev) == "alert"


def test_rename_threshold_separates_a_rename_from_unrelated_columns(base_cfg):
    cols = [("zzz" if c == "notes" else c) for c in COLS if c != "due_date"]
    rows = [[v for c, v in zip(COLS, r) if c != "due_date"] for r in _rows()]
    strict = _events(base_cfg, _table(cols, rows), params=DriftParams(0.99))
    assert {e.kind for e in strict} == {"dropped_optional", "added"}
    assert similarity(base_cfg["baseline"]["notes"], __import__("tenantkit.profile", fromlist=["x"]).profile_column("zzz", [r[-1] for r in rows]), "notes", "zzz") > 0.5


def test_ambiguous_day_month_is_not_reported_as_drift():
    amb = [["01/02/2025"], ["03/04/2025"], ["05/06/2025"]] * 6
    t0 = RawTable("customers.csv", ["customer_id", "created"], [[f"C{i}", r[0]] for i, r in enumerate(amb)], {"kind": "delimited", "encoding": "utf-8"})
    sig = compute_signals([t0])
    cfg = build_config("t", [t0], map_tenant(sig, Params(one_per_entity=False)), sig)["files"]["customers.csv"]
    later = RawTable("customers.csv", ["customer_id", "created"], [[f"C{i}", r[0]] for i, r in enumerate(amb + [["25/12/2025"]])], {"kind": "delimited", "encoding": "utf-8"})
    assert detect_drift(cfg, later) == []


def test_generated_drift_is_found_and_clean_exports_raise_no_alarm(tenants):
    """Every kind of injected change is caught on the fixture tenants, with no false alarms on re-exports."""
    from tenantkit.evaluate import oracle_mapping
    tp = fp = fn = ctrl = 0
    for t in tenants:
        tables = [read_table(p) for p in t.files()]
        sig = compute_signals(tables)
        cfg = build_config(t.tid, tables, oracle_mapping(t.truth()), sig)
        truth = drift_truth_keys(t.truth("drift")["events"])
        found = {e.key() for e in detect_for_tenant(cfg, t, "drift", 0.7)}
        tp += len(found & truth)
        fp += len(found - truth)
        fn += len(truth - found)
        ctrl += len(detect_for_tenant(cfg, t, "control", 0.7))
    assert ctrl == 0 and tp / (tp + fp) > 0.95 and tp / (tp + fn) > 0.85
