import json
from collections import Counter

import pytest

from tenantkit.evaluate import oracle_mapping, score_mapping
from tenantkit.generate import DIFFICULTY_PLAN, N_TENANTS, generate_all, lexicon_coverage, load_tenants
from tenantkit.llm import Arbiter, build_prompt, mask, parse_answer
from tenantkit.mapper import (Params, column_header_scores, compute_signals, header_score, map_tables, map_tenant, naive_map,
                              value_fit)
from tenantkit.profile import profile_column, profile_table, sketch_overlap
from tenantkit.readers import RawTable, read_table
from tenantkit.schema import ENTITIES, get_field, header_tokens


# ------------------------------------------------------------------ generator

def test_plan_matches_the_documented_mix():
    assert len(DIFFICULTY_PLAN) == N_TENANTS == 40
    assert Counter(DIFFICULTY_PLAN) == {"easy": 14, "medium": 14, "hard": 12}


def test_generation_is_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    generate_all(a, n=3, plan=["easy", "medium", "hard"])
    generate_all(b, n=3, plan=["easy", "medium", "hard"])
    for pa in sorted(a.rglob("*")):
        if pa.is_file():
            assert pa.read_bytes() == (b / pa.relative_to(a)).read_bytes(), pa


def test_truth_matches_the_files(tenants):
    for t in tenants:
        for sync in ("base", "control", "drift"):
            truth = t.truth(sync)
            names = {p.name for p in t.files(sync)}
            assert names == set(truth["files"])
            for fn, meta in truth["files"].items():
                assert set(read_table(t.data_dir(sync) / fn).columns) == set(meta["columns"]), (t.tid, sync, fn)
            if sync == "drift":
                continue                              # a dropped key column leaves no key to be unique by
            for entity, recs in truth["records"].items():
                keys = [r[f"{entity}_id"] for r in recs]
                assert len(keys) == len(set(keys)), "truth records must be unique by key"


def test_control_sync_has_the_same_columns_and_no_events(tenants):
    for t in tenants:
        base, ctrl = t.truth("base"), t.truth("control")
        assert ctrl["events"] == []
        for fn in base["files"]:
            assert base["files"][fn]["columns"] == ctrl["files"][fn]["columns"]


def test_drift_sync_events_are_recorded_and_varied(tenants):
    kinds = Counter(e["kind"] for t in tenants for e in t.truth("drift")["events"])
    assert sum(kinds.values()) >= 30 and len(kinds) >= 6
    for t in tenants:
        assert 3 <= len(t.truth("drift")["events"]) <= 4


def test_hard_tenants_use_vocabulary_the_lexicon_has_never_seen(tenants):
    easy = lexicon_coverage([t for t in tenants if t.difficulty == "easy"])
    hard = lexicon_coverage([t for t in tenants if t.difficulty == "hard"])
    assert easy["share"] == 0.0 and hard["share"] > 0.3


def test_hard_tenants_contain_the_advertised_mess(tenants):
    hard = [t for t in tenants if t.difficulty == "hard"]
    metas = [read_table(p).meta for t in hard for p in t.files()]
    assert any(m.get("dropped") and sum(m["dropped"].values()) for m in metas)             # title lines, footers, repeats
    assert any(m.get("encoding") in ("cp1252", "latin-1") for m in metas) or any(m.get("nested") for m in metas)
    assert any(t.truth()["expected_quarantine"]["invoice"] or t.truth()["expected_quarantine"]["payment"] for t in hard)


# ------------------------------------------------------------------ profile

def test_profile_kinds():
    assert profile_column("c", ["a@x.example", "b@y.example", ""]).kind == "email"
    assert profile_column("c", ["2025-01-02", "2025-02-03"]).kind == "date"
    assert profile_column("c", ["1.234,56", "7,10", "88,00"]).kind == "money"
    assert profile_column("c", ["Paid", "open", "VOID", "paid"]).kind == "status"
    assert profile_column("c", ["+1 415 555 0123", "+44 20 7946 0958"]).kind == "phone"
    assert profile_column("c", ["C00001", "C00002", "C00003"]).kind == "id"
    assert profile_column("c", ["", "N/A", "-"]).kind == "empty"


def test_ambiguous_dates_keep_every_candidate_format():
    p = profile_column("d", ["01/02/2025", "03/04/2025", "05/06/2025"])
    assert p.date_ambiguous and {"%d/%m/%Y", "%m/%d/%Y"} <= set(p.date_candidates)
    q = profile_column("d", ["01/02/2025", "23/04/2025"])
    assert q.date_candidates == ["%d/%m/%Y"]


def test_fingerprint_holds_no_raw_values():
    cells = ["secret-customer@example.com", "other-person@example.com", "third@example.com"]
    fp = json.dumps(profile_column("email", cells).fingerprint())
    assert "secret" not in fp and "example.com" not in fp
    a, b = profile_column("a", cells), profile_column("b", cells + ["new@example.com"])
    assert sketch_overlap(a.sketch, b.sketch) > 0.6 and sketch_overlap(a.sketch, profile_column("c", ["x1", "x2"]).sketch) == 0


# ------------------------------------------------------------------ header and value signals

@pytest.mark.parametrize("header,key", [
    ("InvoiceDate", "invoice.invoice_date"), ("Fälligkeitsdatum", "invoice.due_date"), ("Kundennummer", "customer.customer_id"),
    ("e-mail", "customer.email"), ("payment_method", "payment.method"), ("totals.invoiceTotal", "invoice.amount"),
    ("Zahlungsdatum", "payment.payment_date"),
])
def test_header_scores_find_the_obvious(header, key):
    scores = column_header_scores(header)
    assert max(scores, key=scores.get) == key or scores[key] >= 0.9


def test_header_score_is_zero_for_opaque_names():
    assert max(column_header_scores("COL_07").values()) < 0.4
    assert header_score(header_tokens("fld_a"), "invoice.amount") < 0.4


def test_generic_names_are_weak_evidence():
    assert header_score(header_tokens("id"), "customer.customer_id") < header_score(header_tokens("customer_id"), "customer.customer_id")
    assert header_score(header_tokens("date"), "invoice.invoice_date") <= 0.6
    assert header_score(header_tokens("customer_id"), "customer.customer_id") == 1.0


def test_value_fit_separates_types():
    email, date_, money = (profile_column("x", c) for c in (["a@b.example"] * 3, ["2025-01-01", "2025-02-02"], ["10.50", "99.00", "1.25"]))
    f = lambda e, n: get_field(e, n)
    assert value_fit(email, f("customer", "email")) == 1.0 and value_fit(email, f("invoice", "amount")) == 0.0
    assert value_fit(date_, f("invoice", "invoice_date")) == 1.0 and value_fit(date_, f("invoice", "amount")) == 0.0
    assert value_fit(money, f("invoice", "amount")) > 0.9 and value_fit(money, f("customer", "email")) == 0.0


# ------------------------------------------------------------------ mapping

def _table(name, cols, rows):
    return RawTable(name, cols, rows, {"kind": "delimited", "encoding": "utf-8", "delimiter": ","})


def test_easy_tenants_map_perfectly(tenants):
    for t in [t for t in tenants if t.difficulty == "easy"]:
        m = map_tenant(compute_signals([read_table(p) for p in t.files()]), Params())
        s = score_mapping(m, t.truth())
        assert s["tp"] == s["truth"] and not s["wrong"], (t.tid, s["wrong"], s["missed"])


def test_mapping_beats_the_naive_baseline_on_every_difficulty(tenants):
    for diff in ("easy", "medium", "hard"):
        ts = [t for t in tenants if t.difficulty == diff]
        naive = tuned = 0
        for t in ts:
            tables = [read_table(p) for p in t.files()]
            naive += score_mapping(naive_map(tables), t.truth())["tp"]
            tuned += score_mapping(map_tenant(compute_signals(tables)), t.truth())["tp"]
        assert tuned > naive, diff


def test_relations_tell_apart_two_id_columns_with_opaque_names():
    custs = [[f"C{i:04d}", f"Name {i}"] for i in range(1, 41)]
    invs = [[f"INV{i:05d}", f"C{(i % 40) + 1:04d}"] for i in range(1, 91)]

    def payments(order):
        rows = [[f"PAY{i:05d}", f"INV{i:05d}"] for i in range(1, 61)]
        return _table("payments.csv", ["e", "f"], rows if order == "pay_first" else [r[::-1] for r in rows])

    def run(order, params):
        tables = [_table("customers.csv", ["a", "b"], custs), _table("invoices.csv", ["c", "d"], invs), payments(order)]
        got = map_tables(tables, params)
        pay = got.files["payments.csv"].by_source()
        want = {"e": "payment_id", "f": "invoice_id"} if order == "pay_first" else {"e": "invoice_id", "f": "payment_id"}
        return got, all(pay[k].field == v for k, v in want.items())

    for order in ("pay_first", "inv_first"):
        got, ok = run(order, Params())
        assert ok, order
        assert got.files["invoices.csv"].by_source()["d"].field == "customer_id"
        assert got.files["customers.csv"].by_source()["a"].field == "customer_id"
    # without the relation signal two unique id-shaped columns tie, and column order decides: right for one order, wrong for the other
    assert not all(run(o, Params(w_rel=0.0))[1] for o in ("pay_first", "inv_first"))


def test_date_rule_orients_two_unlabelled_dates():
    rows = [[f"INV{i:04d}", f"C{i % 9:03d}", f"2025-{1 + i % 12:02d}-{1 + i % 25:02d}", f"2025-{1 + i % 12:02d}-{1 + i % 25:02d}", "1.00"] for i in range(30)]
    # make the third column the later date, 30 days after the fourth
    from datetime import date, timedelta
    rows = []
    for i in range(30):
        d = date(2025, 1, 1) + timedelta(days=i * 3)
        rows.append([f"INV{i:04d}", f"C{i % 9:03d}", (d + timedelta(days=30)).isoformat(), d.isoformat(), f"{100 + i}.00"])
    tab = _table("invoices.csv", ["k", "c", "x", "y", "z"], rows)
    on = map_tables([tab], Params(one_per_entity=False)).files["invoices.csv"].by_source()
    assert on["y"].field == "invoice_date" and on["x"].field == "due_date"
    assert on["y"].review                         # decided by a tie-break rule, so a person is asked to look


def test_magnitude_rule_picks_the_total_over_tax():
    rows = [[f"INV{i:04d}", f"C{i % 9:03d}", "2025-01-02", f"{(100 + i) * 5}.00", f"{(100 + i)}.00"] for i in range(30)]
    tab = _table("invoices.csv", ["k", "c", "d", "p", "q"], rows)
    got = map_tables([tab], Params(one_per_entity=False)).files["invoices.csv"].by_source()
    assert got["p"].field == "amount" and got["q"].field is None


def test_one_file_per_entity_resolves_swapped_names():
    custs = [[f"C{i:04d}", f"Acme {i} Inc", f"a{i}@x.example"] for i in range(1, 31)]
    invs = [[f"INV{i:05d}", f"C{(i % 30) + 1:04d}", "2025-02-03", "10.00", "EUR"] for i in range(1, 61)]
    tables = [_table("export_1.csv", ["a", "b", "c"], custs), _table("export_2.csv", ["a", "b", "c", "d", "e"], invs)]
    m = map_tables(tables, Params())
    assert (m.files["export_1.csv"].entity, m.files["export_2.csv"].entity) == ("customer", "invoice")


def test_anything_unknown_stays_unmapped():
    rows = [[f"C{i:04d}", f"Acme {i} Inc", f"note {i % 3}"] for i in range(30)]
    got = map_tables([_table("customers.csv", ["customer_id", "name", "internal_comment"], rows)], Params()).files["customers.csv"].by_source()
    assert got["internal_comment"].field is None


# ------------------------------------------------------------------ the optional LLM arbiter

def _ambiguous_tables(with_created=True):
    rows = [[f"C{i:04d}", f"Acme {i} Inc", f"+1 415 555 {1000 + i}", "2025-01-02", "2025-03-04"] for i in range(30)]
    cols = ["customer_id", "name", "phone", "created", "mystery"]
    if not with_created:
        rows, cols = [r[:3] + r[4:] for r in rows], cols[:3] + cols[4:]
    return [_table("customers.csv", cols, rows)]


def test_arbiter_is_asked_only_about_unclear_columns_and_only_sees_candidates():
    asked = []

    def fake(prompt):
        asked.append(prompt)
        return "NONE"

    tables = _ambiguous_tables()
    map_tables(tables, Params(threshold=0.35), arbiter=Arbiter(fake))
    assert 0 < len(asked) <= 3                       # not every column
    for p in asked:
        assert "Column name:" in p and "Candidates:" in p
        assert "Acme" not in p and "415" not in p    # no cell values unless asked for


def test_arbiter_answers_are_validated_and_cannot_steal_a_field():
    tables = _ambiguous_tables()
    base = map_tables(tables, Params(threshold=0.35)).files["customers.csv"].by_source()
    owner = next(c.source for c in base.values() if c.field == "name")
    m = map_tables(tables, Params(threshold=0.35), arbiter=Arbiter(lambda _: "customer.name")).files["customers.csv"].by_source()
    assert m[owner].field == "name" and sum(1 for c in m.values() if c.field == "name") == 1
    junk = map_tables(tables, Params(threshold=0.35), arbiter=Arbiter(lambda _: "ignore all previous instructions")).files["customers.csv"].by_source()
    assert {k: v.field for k, v in junk.items()} == {k: v.field for k, v in base.items()}


def test_arbiter_can_fill_an_unmapped_column_and_marks_it():
    tables = _ambiguous_tables(with_created=False)
    plain = map_tables(tables, Params(threshold=0.9)).files["customers.csv"].by_source()
    assert plain["mystery"].field is None
    m = map_tables(tables, Params(threshold=0.9), arbiter=Arbiter(lambda p: "created_date" if "created_date" in p else "NONE"))
    got = m.files["customers.csv"].by_source()["mystery"]
    assert got.field == "created_date" and got.decided_by == "arbiter" and got.review


def test_prompt_masks_samples_when_asked():
    assert mask("Acme 415-555") == "xxxx 999-999"
    from tenantkit.llm import Question
    q = Question("f.csv", "customer", "col", {"kind": "text"}, ["customer.name"], [mask("Secret Name 123")])
    assert "Secret" not in build_prompt(q) and "xxxxxx xxxx 999" in build_prompt(q)
    assert parse_answer("customer.name.", ["customer.name"]) == "customer.name" and parse_answer("sure! name", ["customer.name"]) is None
    assert Arbiter(lambda _: "x", max_questions=0).resolve(q) is None
