import json
from datetime import date
from decimal import Decimal

import pytest

from tenantkit.parsing import (decimal_vote, id_shaped, infer_decimal, looks_like_phone, normalize_phone, parse_date, parse_money,
                               shape, split_money)
from tenantkit.readers import decode, read_delimited, read_json, read_table, sniff_delimiter


# ------------------------------------------------------------------ parsing

@pytest.mark.parametrize("fmt,text,want", [
    ("%Y-%m-%d", "2025-03-04", date(2025, 3, 4)),
    ("%d/%m/%Y", "04/03/2025", date(2025, 3, 4)),
    ("%m/%d/%Y", "3/4/2025", date(2025, 3, 4)),
    ("%d.%m.%Y", "04.03.2025", date(2025, 3, 4)),
    ("%d-%b-%Y", "04-Mar-2025", date(2025, 3, 4)),
    ("%b %d, %Y", "Mar 04, 2025", date(2025, 3, 4)),
    ("%Y%m%d", "20250304", date(2025, 3, 4)),
    ("%Y-%m-%dT%H:%M:%S", "2025-03-04T09:30:00Z", date(2025, 3, 4)),
    ("epoch", "1741080600", date(2025, 3, 4)),
])
def test_parse_date_formats(fmt, text, want):
    assert parse_date(text, fmt) == want


def test_parse_date_rejects_impossible_and_implausible():
    assert parse_date("31/02/2025", "%d/%m/%Y") is None
    assert parse_date("00900001", "%Y%m%d") is None            # a zero-padded id, not 0090-00-01
    assert parse_date("13/13/2025", "%m/%d/%Y") is None


@pytest.mark.parametrize("raw,decimal,want,cur", [
    ("1234.56", ".", "1234.56", None),
    ("1,234.56", ".", "1234.56", None),
    ("1.234,56", ",", "1234.56", None),
    ("1 234,56", ",", "1234.56", None),
    ("1'234.56", ".", "1234.56", None),
    ("EUR 1.234,56", ",", "1234.56", "EUR"),
    ("1.234,56 €", ",", "1234.56", "EUR"),
    ("$1,234.56", ".", "1234.56", "USD"),
    ("(12.50)", ".", "-12.50", None),
    ("1234.50 GBP", ".", "1234.50", "GBP"),
])
def test_parse_money(raw, decimal, want, cur):
    amount, currency = parse_money(raw, decimal)
    assert amount == Decimal(want) and currency == cur


def test_parse_money_scale_and_garbage():
    assert parse_money("123456", ".", Decimal("0.01"))[0] == Decimal("1234.56")
    assert parse_money("TBD")[0] is None
    assert parse_money("")[0] is None


def test_decimal_inference_votes_across_a_column():
    assert infer_decimal([split_money(v)[0] for v in ["1.234,56", "89,10", "7,00"]]) == ","
    assert infer_decimal([split_money(v)[0] for v in ["1,234.56", "89.10"]]) == "."
    assert decimal_vote("1,234") is None            # could be either, so it casts no vote


def test_phone_and_id_are_not_confused():
    assert looks_like_phone("+1 (415) 555-0123") and looks_like_phone("+14155550123") and looks_like_phone("001 415 555 0123")
    assert not looks_like_phone("0040012")          # zero-padded invoice number
    assert not looks_like_phone("00900001")
    assert normalize_phone("+1 (415) 555-0123") == "+14155550123"
    assert normalize_phone("001 415 555 0123") == "+14155550123"
    assert normalize_phone("415 555 0123") is None  # no country code: cannot be made international
    assert id_shaped("INV-2025-0042") and id_shaped("C00123") and not id_shaped("12.50") and not id_shaped("hello world")


def test_shape_collapses_runs():
    assert shape("C-00123") == "A-9" and shape("INV2025-0042") == "A9-9"


# ------------------------------------------------------------------ readers

def test_decode_prefers_utf8_then_cp1252():
    assert decode("Müller".encode("utf-8")) == ("Müller", "utf-8")
    assert decode(b"\xef\xbb\xbfabc") == ("abc", "utf-8-sig")
    text, enc = decode("Preis 10 € Währung".encode("cp1252"))
    assert enc == "cp1252" and text == "Preis 10 € Währung"


def test_delimited_junk_is_skipped():
    raw = ("Export from LegacyERP v4.2\r\nGenerated: 2025-11-03\r\n\r\n"
           "id;name;amount\r\n1;Acme, Inc.;10,50\r\n2;Bolt GmbH;7,00\r\n\r\nid;name;amount\r\n3;Cog AG;1,00\r\nTotal records: 3\r\n;;\r\n")
    t = read_delimited(raw.encode("cp1252"))
    assert t.columns == ["id", "name", "amount"] and len(t.rows) == 3
    assert t.rows[0] == ["1", "Acme, Inc.", "10,50"]
    d = t.meta["dropped"]
    assert d["title_lines"] == 3 and d["repeated_header"] == 1 and d["footer"] == 1 and d["blank"] >= 2
    assert t.meta["delimiter"] == ";"


@pytest.mark.parametrize("d", [",", ";", "\t", "|"])
def test_delimiter_sniffing(d):
    text = "\n".join(d.join(r) for r in [["a", "b", "c"], ["1", "2", "3"], ["4", "5", "6"]])
    assert sniff_delimiter(text) == d


def test_duplicate_headers_are_made_unique():
    t = read_delimited(b"a,a,b\n1,2,3\n")
    assert t.columns == ["a", "a_2", "b"]


def test_nested_and_wrapped_json_is_flattened():
    payload = {"meta": {"count": 2}, "data": [{"id": 1, "contact": {"email": "a@x.example"}, "ok": True},
                                              {"id": 2, "contact": {"email": None}, "ok": False}]}
    t = read_json(json.dumps(payload).encode())
    assert t.columns == ["id", "contact.email", "ok"] and t.rows == [["1", "a@x.example", "true"], ["2", "", "false"]]
    nd = read_json(b'{"a": 1}\n{"a": 2, "b": 3}\n', ndjson=True)
    assert nd.columns == ["a", "b"] and nd.rows[0] == ["1", ""]


def test_excel_is_refused_clearly(tmp_path):
    p = tmp_path / "x.xlsx"
    p.write_bytes(b"PK")
    with pytest.raises(ValueError, match="Excel"):
        read_table(p)


def test_every_generated_file_reads_with_the_true_headers(tenants):
    for t in tenants:
        truth = t.truth()
        for p in t.files():
            tab = read_table(p)
            assert set(tab.columns) == set(truth["files"][p.name]["columns"]), (t.tid, p.name)
            assert len(tab.rows) >= len(truth["records"][truth["files"][p.name]["entity"]])
