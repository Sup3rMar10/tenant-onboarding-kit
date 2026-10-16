import csv
import json
from datetime import date, timedelta

from tenantkit.config import build_config
from tenantkit.ingest import Ledger, ingest_tenant
from tenantkit.mapper import Params, compute_signals, map_tenant
from tenantkit.readers import read_table


def test_healed_drift_is_loaded_and_flagged(tmp_path):
    def write(name, header, rows):
        with open(tmp_path / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)
        return tmp_path / name

    custs = [[f"C{i:04d}", f"Acme {i} Inc"] for i in range(1, 21)]

    def invoices(fmt, extra=0):
        out = []
        for i in range(40 + extra):
            d = date(2025, 1, 1) + timedelta(days=i * 4)
            out.append([f"INV{50000 + i}", f"C{(i % 20) + 1:04d}", d.strftime(fmt), (d + timedelta(days=30)).strftime(fmt), "EUR", f"{100 + i * 7}.50", "open"])
        return out

    head = ["invoice_id", "customer_id", "invoice_date", "due_date", "currency", "amount", "status"]
    f1 = [write("customers.csv", ["customer_id", "name"], custs), write("invoices.csv", head, invoices("%Y-%m-%d"))]
    tables = [read_table(p) for p in f1]
    sig = compute_signals(tables)
    cfg = build_config("mini", tables, map_tenant(sig, Params()), sig)
    led = Ledger()
    assert ingest_tenant(led, cfg, f1).status == "committed"
    # next export: one column renamed, and the dates now arrive as dd.mm.yyyy
    f2 = [write("customers.csv", ["customer_id", "name"], custs),
          write("invoices.csv", ["invoice_id", "customer_id", "Belegdatum", "due_date", "currency", "amount", "status"], invoices("%d.%m.%Y", extra=5))]
    res = ingest_tenant(led, cfg, f2)
    assert res.status == "committed" and res.healed
    assert {e.kind for e in res.drift} == {"renamed", "retyped_format"} and all(e.action == "heal" for e in res.drift)
    rows = [json.loads(r[0]) for r in led.db.execute("SELECT data FROM records WHERE entity='invoice'")]
    assert len(rows) == 45 and all(r["invoice_date"].startswith("2025-") and r["due_date"].startswith("2025-") for r in rows)
