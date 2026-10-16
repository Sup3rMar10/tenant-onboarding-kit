import copy
import json

import pytest

from tenantkit.config import ConfigError, build_config, load_config, save_config, validate_config
from tenantkit.evaluate import aggregate_normalization, oracle_mapping, score_normalization
from tenantkit.mapper import Params, compute_signals, map_tenant
from tenantkit.normalize import normalize_table
from tenantkit.readers import RawTable, read_delimited, read_table
from tenantkit.validate import validate


def _cfg_for(tenant, oracle=True):
    tables = [read_table(p) for p in tenant.files()]
    sig = compute_signals(tables)
    mapping = oracle_mapping(tenant.truth()) if oracle else map_tenant(sig, Params())
    return build_config(tenant.tid, tables, mapping, sig), tables


def _file_cfg(entity, cols):
    return {"entity": entity, "columns": [{"source": s, "target": t, "transform": tr} for s, t, tr in cols], "baseline": {}}


def _tab(cols, rows, name="x.csv"):
    return RawTable(name, cols, rows, {"kind": "delimited", "encoding": "utf-8", "dropped": {}})


# ------------------------------------------------------------------ config as code

def test_config_round_trips_and_is_reviewable(easy_tenant, tmp_path):
    cfg, _ = _cfg_for(easy_tenant)
    assert validate_config(cfg) == []
    path = tmp_path / "c.json"
    save_config(cfg, path)
    assert load_config(path) == json.loads(path.read_text(encoding="utf-8"))
    some = next(iter(cfg["files"].values()))
    assert {"entity", "format", "columns", "baseline", "required_unmapped"} <= set(some)
    mapped = [c for c in some["columns"] if c["target"]]
    assert all("transform" in c and "confidence" in c and "needs_review" in c for c in mapped)


def test_validation_says_what_is_wrong_in_words(easy_tenant):
    cfg, _ = _cfg_for(easy_tenant)
    fn = next(n for n in cfg["files"] if n.startswith("customers"))

    def edited(fn_edit):
        bad = copy.deepcopy(cfg)
        fn_edit(bad)
        return " | ".join(validate_config(bad))

    def col(bad, target):
        return next(c for c in bad["files"][fn]["columns"] if c["target"] == target)

    assert "not a customer field" in edited(lambda b: col(b, "email").update(target="no_such_field"))
    assert "is mapped from both" in edited(lambda b: col(b, "email").update(target="name"))
    assert "required field 'customer_id' is not mapped" in edited(lambda b: col(b, "customer_id").update(target=None))
    assert "not one of" in edited(lambda b: b["drift_policy"].update(renamed="ignore"))
    assert "unknown date format" in edited(lambda b: col(b, "created_date")["transform"].update(format="%Q"))
    # allow_missing is the explicit way to say "this customer really has no such column"
    ok = copy.deepcopy(cfg)
    next(c for c in ok["files"][fn]["columns"] if c["target"] == "customer_id")["target"] = None
    ok["files"][fn]["allow_missing"] = ["customer_id"]
    assert validate_config(ok) == []


def test_load_config_raises_with_every_problem(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"config_version": 9, "files": {"a.csv": {"entity": "widget", "columns": []}}}))
    with pytest.raises(ConfigError) as e:
        load_config(p)
    assert any("config_version" in x for x in e.value.problems) and any("widget" in x for x in e.value.problems)


def test_a_human_edit_changes_what_is_loaded(tenants):
    easy = next(t for t in tenants if any(m["columns"] and any(v == "invoice.status" for v in m["columns"].values())
                                          for m in t.truth()["files"].values()) and t.difficulty == "easy")
    cfg, tables = _cfg_for(easy)
    t = next(t for t in tables if t.name.startswith("invoices"))
    fc = copy.deepcopy(cfg["files"][t.name])
    before = normalize_table(t, fc)
    for c in fc["columns"]:
        if c["target"] == "status":
            c["transform"]["value_map"] = {k: "void" for k in c["transform"]["value_map"]}
    after = normalize_table(t, fc)
    assert {r["status"] for r in after.records} <= {"void", None} and "void" in {r["status"] for r in after.records}
    assert {r["status"] for r in before.records} - {"void", None}


# ------------------------------------------------------------------ transforms

def test_each_transform():
    cfg = _file_cfg("invoice", [
        ("id", "invoice_id", {"type": "id"}), ("cust", "customer_id", {"type": "id"}),
        ("d", "invoice_date", {"type": "date", "format": "%d.%m.%Y"}), ("due", "due_date", {"type": "date", "format": "%d.%m.%Y"}),
        ("amt", "amount", {"type": "money", "decimal": ",", "scale": 1, "currency_in_value": True}),
        ("st", "status", {"type": "status", "value_map": {"offen": "open", "bezahlt": "paid"}, "unmapped": []})])
    t = _tab(["id", "cust", "d", "due", "amt", "st"], [["I1", "C1", "04.03.2025", "03.04.2025", "EUR 1.234,56", "Offen"],
                                                      ["I2", "C1", "05.03.2025", "", "10,00 EUR", "weird"]])
    out = normalize_table(t, cfg)
    r1, r2 = out.records
    assert r1 == {"invoice_id": "I1", "customer_id": "C1", "invoice_date": "2025-03-04", "due_date": "2025-04-03", "currency": "EUR",
                  "amount": "1234.56", "status": "open", "_row": 1}
    assert r2["due_date"] is None and r2["status"] is None and r2["amount"] == "10.00"
    assert [(i.code, i.column) for i in out.issues] == [("unknown_status", "status")]


def test_cents_scale_phone_country_email():
    cfg = _file_cfg("customer", [("id", "customer_id", {"type": "id"}), ("n", "name", {"type": "text"}), ("e", "email", {"type": "email"}),
                                 ("p", "phone", {"type": "phone"}), ("c", "country", {"type": "country"})])
    t = _tab(["id", "n", "e", "p", "c"], [["1", "A Inc", "Bill@X.EXAMPLE", "001 415 555 0123", "Deutschland"], ["2", "B", "nope", "415 555", "Atlantis"]])
    a, b = normalize_table(t, cfg).records
    assert (a["email"], a["phone"], a["country"]) == ("bill@x.example", "+14155550123", "DE")
    assert (b["email"], b["phone"], b["country"]) == (None, None, None)
    inv = _file_cfg("payment", [("id", "payment_id", {"type": "id"}), ("i", "invoice_id", {"type": "id"}),
                                ("d", "payment_date", {"type": "date", "format": "%Y-%m-%d"}),
                                ("a", "amount", {"type": "money", "decimal": ".", "scale": 0.01})])
    out = normalize_table(_tab(["id", "i", "d", "a"], [["p1", "i1", "2025-01-02", "123456"]]), inv)
    assert out.records[0]["amount"] == "1234.56"


def test_bad_required_values_are_quarantined_not_loaded():
    cfg = _file_cfg("payment", [("id", "payment_id", {"type": "id"}), ("i", "invoice_id", {"type": "id"}),
                                ("d", "payment_date", {"type": "date", "format": "%Y-%m-%d"}), ("a", "amount", {"type": "money", "decimal": "."})])
    out = normalize_table(_tab(["id", "i", "d", "a"], [["p1", "i1", "2025-01-02", "10.00"], ["p2", "i1", "2025-01-02", "TBD"],
                                                      ["p3", "", "2025-01-02", "5.00"], ["p4", "i2", "not a date", "5.00"]]), cfg)
    assert [r["payment_id"] for r in out.records] == ["p1"]
    assert [q[0] for q in out.quarantined] == [2, 3, 4]
    assert "unparseable_amount:amount" in out.quarantined[0][1] and "missing_required:invoice_id" in out.quarantined[1][1]


def test_duplicate_keys_last_wins_and_are_counted():
    cfg = _file_cfg("customer", [("id", "customer_id", {"type": "id"}), ("n", "name", {"type": "text"})])
    out = normalize_table(_tab(["id", "n"], [["1", "A"], ["1", "A"], ["2", "B"], ["2", "B changed"]]), cfg)
    assert {r["customer_id"]: r["name"] for r in out.records} == {"1": "A", "2": "B changed"}
    assert out.stats["duplicates_identical"] == 1 and out.stats["duplicates_conflicting"] == 1 and out.stats["loaded"] == 2


def test_normalization_with_the_true_mapping_is_accurate_on_easy_and_medium(tenants):
    scores = []
    for t in [t for t in tenants if t.difficulty != "hard"]:
        cfg, tables = _cfg_for(t)
        norm = {tb.name: normalize_table(tb, cfg["files"][tb.name]) for tb in tables}
        scores.append(score_normalization(norm, t.truth()))
    agg = aggregate_normalization(scores)
    assert agg["record_recall"] == 1.0 and agg["false_loads"] == 0 and agg["field_accuracy"] > 0.97


def test_junk_amount_rows_are_held_back(tenants):
    scores = []
    for t in [t for t in tenants if t.difficulty == "hard"]:
        cfg, tables = _cfg_for(t)
        norm = {tb.name: normalize_table(tb, cfg["files"][tb.name]) for tb in tables}
        scores.append(score_normalization(norm, t.truth()))
    agg = aggregate_normalization(scores)
    assert agg["junk_rows_held_back"] == 1.0 and agg["false_loads"] == 0


# ------------------------------------------------------------------ data-quality report

def _norm(tenant, oracle=True):
    cfg, tables = _cfg_for(tenant, oracle)
    return cfg, {tb.name: normalize_table(tb, cfg["files"][tb.name]) for tb in tables}


def test_clean_tenant_report_and_no_raw_values(easy_tenant):
    cfg, norm = _norm(easy_tenant)
    rep = validate(easy_tenant.tid, norm)
    text = rep.render()
    assert rep.decision in ("ok", "warn") and "Decision:" in text
    emails = [r["email"] for nf in norm.values() for r in nf.records if r.get("email")]
    names = [r["name"] for nf in norm.values() for r in nf.records if r.get("name")]
    assert emails and all(e not in text for e in emails) and all(n not in text for n in names)
    assert json.dumps(rep.to_dict()) and all(e not in json.dumps(rep.to_dict()) for e in emails)


def test_a_wrong_mapping_is_blocked_by_the_report(easy_tenant):
    cfg, tables = _cfg_for(easy_tenant)
    bad = copy.deepcopy(cfg)
    fn = next(n for n in bad["files"] if n.startswith("invoices"))
    for c in bad["files"][fn]["columns"]:
        if c["target"] == "amount":
            c["transform"] = {"type": "date", "format": "%Y-%m-%d"}          # pretend the amount is a date
    norm = {tb.name: normalize_table(tb, bad["files"][tb.name]) for tb in tables}
    rep = validate(easy_tenant.tid, norm)
    assert rep.decision == "block" and any(f.code == "quarantine_rate" for f in rep.findings)


def test_orphan_references_and_unmapped_required_block(easy_tenant):
    cfg, norm = _norm(easy_tenant)
    inv = next(n for n in norm if n.startswith("invoices"))
    for r in norm[inv].records:
        r["customer_id"] = "NOPE-" + r["customer_id"]
    rep = validate(easy_tenant.tid, norm)
    assert rep.decision == "block" and any(f.code == "orphan_references" for f in rep.findings)
    cfg2, tables = _cfg_for(easy_tenant)
    fn = next(n for n in cfg2["files"] if n.startswith("customers"))
    for c in cfg2["files"][fn]["columns"]:
        if c["target"] == "name":
            c["target"] = None
    norm2 = {tb.name: normalize_table(tb, cfg2["files"][tb.name]) for tb in tables}
    assert any(f.code == "unmapped_required" for f in validate(easy_tenant.tid, norm2).findings)


def test_existing_keys_count_as_parents(easy_tenant):
    cfg, norm = _norm(easy_tenant)
    existing = {"customer": {r["customer_id"] for nf in norm.values() if nf.entity == "customer" for r in nf.records}}
    only_invoices = {k: v for k, v in norm.items() if v.entity == "invoice"}
    assert validate(easy_tenant.tid, only_invoices).decision == "ok" or all(f.code != "orphan_references" for f in validate(easy_tenant.tid, only_invoices, existing).findings)


def test_build_config_matches_columns_by_name_not_position():
    """Regression: a mapping that lists columns in a different order than the file (nested JSON groups them)
    must not get another column's transform."""
    from tenantkit.mapper import ColumnMapping, FileMapping, TenantMapping

    tab = _tab(["meta.created", "id", "name"], [["25.12.2024", f"C{i:03d}", f"Acme {i} Inc"] for i in range(20)], name="customers.json")
    sig = compute_signals([tab])
    mapping = TenantMapping({"customers.json": FileMapping("customers.json", "customer", {}, [
        ColumnMapping("id", "customer_id"), ColumnMapping("name", "name"), ColumnMapping("meta.created", "created_date")])})
    cfg = build_config("t", [tab], mapping, sig)
    cols = {c["source"]: c for c in cfg["files"]["customers.json"]["columns"]}
    assert cols["meta.created"]["transform"] == {"type": "date", "format": "%d.%m.%Y", "ambiguous": False}
    assert cols["id"]["transform"] == {"type": "id"} and cols["name"]["transform"] == {"type": "text"}
    assert normalize_table(tab, cfg["files"]["customers.json"]).records[0]["created_date"] == "2024-12-25"
