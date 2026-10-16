"""A seeded generator of synthetic customer data exports ("tenants").

Each tenant is one pretend customer that exports customers, invoices and payments
from its own system. The exports disagree about almost everything: header names
and languages, file format, delimiter, encoding, date and number formats, nested
or flat layout, junk rows, duplicate keys, and which optional columns exist.

Ground truth is written next to the data (truth.json): which source column is which
canonical field, and what the records should look like after normalization. The
mapper never reads truth.json. Only evaluate.py does.

Header vocabulary has two tiers. "core" is the kind of thing the mapper's lexicon
(schema.py) was written to cover. "tail" is vocabulary the lexicon has never seen
(other languages, internal jargon, abbreviations), so the mapper cannot simply
look it up. `lexicon_coverage` measures how much of the generated headers the
lexicon knows, so the effect is reported rather than hidden.

Everything here is synthetic. No real customer, invoice or person is involved.
"""
from __future__ import annotations

import csv
import io
import json
import random
import re
import shutil
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from .schema import COUNTRIES, ENTITIES

GEN_SEED = 7
N_TENANTS = 40
DIFFICULTY_PLAN = ["easy"] * 14 + ["medium"] * 14 + ["hard"] * 12

# ------------------------------------------------------------------ vocabulary

HEADERS: dict[str, dict[str, list[str]]] = {
    "customer.customer_id": {
        "core": ["customer_id", "customer number", "cust_no", "client id", "account number", "Kundennummer", "KundenNr"],
        "tail": ["kd_nr", "acct", "debtor_no", "partner_ref", "n_cliente", "klantnummer", "cod_cliente", "DebitorNr", "customer_key"]},
    "customer.name": {
        "core": ["name", "customer name", "company", "company name", "client name", "Firma", "nombre", "organization"],
        "tail": ["bill_to_name", "debtor", "klantnaam", "ragione_sociale", "account_title", "legal_entity", "naam", "nome_cliente", "party_name"]},
    "customer.email": {
        "core": ["email", "e-mail", "email address", "contact email", "correo", "mail"],
        "tail": ["ap_contact", "kontakt_mail", "courriel_contact", "billing_contact_email", "posta_elettronica", "emailadres", "remit_to_email"]},
    "customer.phone": {
        "core": ["phone", "telephone", "tel", "phone number", "telefon", "mobile"],
        "tail": ["tel_no", "contact_number", "telefono_fijo", "telefoonnummer", "fono", "voice", "cell"]},
    "customer.country": {
        "core": ["country", "country_code", "land", "pais", "pays"],
        "tail": ["ctry_cd", "nation_iso", "paese", "land_code", "territory", "country_of_residence"]},
    "customer.created_date": {
        "core": ["created", "created_at", "customer_since", "signup_date", "Erstellt", "fecha_alta"],
        "tail": ["onboard_dt", "creation_ts", "opened_on", "aangemaakt", "data_creazione", "reg_date", "dt_created", "kunde_seit"]},
    "invoice.invoice_id": {
        "core": ["invoice_id", "invoice_no", "invoice number", "Rechnungsnummer", "factura", "inv_no", "document number"],
        "tail": ["belegnr", "doc_ref", "fattura_n", "factuurnummer", "bill_ref", "voucher_no", "ar_doc", "invoice_key", "documento"]},
    "invoice.customer_id": {
        "core": ["customer_id", "customer", "cust_id", "client", "Kundennummer", "id_cliente"],
        "tail": ["bill_to", "sold_to", "debtor_ref", "kd_nr", "klant", "party", "account_ref", "cod_cliente"]},
    "invoice.invoice_date": {
        "core": ["invoice_date", "date", "issued", "issue_date", "Rechnungsdatum", "fecha_factura", "inv_date"],
        "tail": ["doc_date", "posted", "booking_date", "data_fattura", "factuurdatum", "dt_emissao", "issued_dt", "raised_on", "txn_dt"]},
    "invoice.due_date": {
        "core": ["due_date", "due", "payable_by", "Fälligkeitsdatum", "vencimiento", "maturity"],
        "tail": ["net_date", "scadenza", "vervaldatum", "pay_by_dt", "terms_date", "settle_by", "due_dt", "zahlbar_bis"]},
    "invoice.currency": {
        "core": ["currency", "curr", "ccy", "Währung", "moneda"],
        "tail": ["cur_cd", "valuta", "munt", "iso_ccy", "money_code", "waehrung_code", "mon"]},
    "invoice.amount": {
        "core": ["amount", "total", "invoice_total", "Betrag", "importe", "gross_amount", "amt"],
        "tail": ["grand_amt", "totale", "bedrag", "valor", "invoice_value", "net_plus_tax", "amount_due_total", "sum_gross"]},
    "invoice.status": {
        "core": ["status", "state", "invoice_status", "Status", "estado", "payment_status"],
        "tail": ["stat", "dunning_state", "stato", "toestand", "situacion", "pay_state", "st", "open_closed_flag"]},
    "payment.payment_id": {
        "core": ["payment_id", "payment_no", "receipt_no", "transaction_id", "Zahlungsnummer", "txn_id"],
        "tail": ["remit_ref", "pmt_key", "bankref", "betaling_id", "pago_ref", "cash_app_id", "settle_id", "rcpt"]},
    "payment.invoice_id": {
        "core": ["invoice_id", "invoice", "invoice_no", "applied_to", "Rechnungsnummer", "factura"],
        "tail": ["applied_inv", "settles_doc", "fattura_rif", "factuur_ref", "doc_applied", "clearing_doc", "against", "bill_ref"]},
    "payment.payment_date": {
        "core": ["payment_date", "paid_on", "date_paid", "received", "Zahlungsdatum", "fecha_pago", "txn_date"],
        "tail": ["value_dt", "booked", "data_pagamento", "betaaldatum", "clearing_date", "rcpt_dt", "settled_on", "cash_date"]},
    "payment.amount": {
        "core": ["amount", "payment_amount", "paid", "amount_paid", "Betrag", "importe", "amt"],
        "tail": ["pmt_value", "incasso", "betaald_bedrag", "remitted", "cleared_amt", "received_total", "importo", "net_receipt"]},
    "payment.method": {
        "core": ["method", "payment_method", "type", "Zahlungsart", "metodo", "mode", "channel"],
        "tail": ["tender_type", "pmt_mode", "modalita", "betaalwijze", "rail", "instrument", "forma_pgto", "how_paid"]},
}
CENTS_HEADERS = ["amount_cents", "total_cents", "betrag_cent", "importe_centimos"]
EXTRA_HEADERS = {
    "sales_rep": ["sales_rep", "account_manager", "owner", "rep"],
    "region": ["region", "region_code", "territory_code", "area"],
    "segment": ["segment", "tier", "customer_type", "class"],
    "legacy_flag": ["legacy_flag", "migrated", "is_active", "flag_a"],
    "updated_at": ["updated_at", "last_modified", "modified", "last_update"],
    "notes": ["notes", "comment", "remarks", "memo"],
    "tax_amount": ["tax_amount", "vat", "tax", "mwst"],
    "balance": ["balance", "open_balance", "outstanding", "rest"],
    "po_number": ["po_number", "po", "purchase_order", "customer_po"],
    "terms": ["terms", "payment_terms", "net_terms", "zahlungsziel"],
    "bank_ref": ["bank_ref", "bank_reference", "statement_ref", "bank_txn"],
    "batch_id": ["batch_id", "batch", "import_batch", "run_id"],
    "currency_col": ["currency", "ccy", "curr", "cur"],
    "row_hash": ["row_hash", "checksum", "hash", "etag"],
}
EXTRAS_BY_ENTITY = {
    "customer": ["sales_rep", "region", "segment", "legacy_flag", "updated_at", "notes", "row_hash"],
    "invoice": ["tax_amount", "balance", "po_number", "terms", "updated_at", "notes", "batch_id"],
    "payment": ["bank_ref", "currency_col", "batch_id", "updated_at", "notes", "row_hash"],
}
EASY_EXTRAS = {"sales_rep", "region", "segment", "notes", "terms", "batch_id"}

STATUS_WORDS = {
    "en": {"open": "Open", "paid": "Paid", "overdue": "Overdue", "void": "Void", "partial": "Partial"},
    "de": {"open": "offen", "paid": "bezahlt", "overdue": "überfällig", "void": "storniert", "partial": "teilbezahlt"},
    "es": {"open": "pendiente", "paid": "pagada", "overdue": "vencida", "void": "anulada", "partial": "parcial"},
    "fr": {"open": "ouvert", "paid": "payé", "overdue": "en retard", "void": "annulé", "partial": "partiel"},
    "it": {"open": "aperta", "paid": "pagata", "overdue": "scaduta", "void": "annullata", "partial": "parziale"},
    "nl": {"open": "open", "paid": "betaald", "overdue": "achterstallig", "void": "geannuleerd", "partial": "gedeeltelijk"},
    "codes": {"open": "A", "paid": "C", "overdue": "L", "void": "X", "partial": "S"},
}
METHOD_WORDS = {
    "en": {"card": "Credit Card", "ach": "ACH", "wire": "Wire Transfer", "check": "Check", "cash": "Cash"},
    "de": {"card": "Kreditkarte", "ach": "Lastschrift", "wire": "Überweisung", "check": "Scheck", "cash": "Bargeld"},
    "es": {"card": "Tarjeta", "ach": "Domiciliación", "wire": "Transferencia", "check": "Cheque", "cash": "Efectivo"},
    "fr": {"card": "Carte", "ach": "Prélèvement", "wire": "Virement", "check": "Chèque", "cash": "Espèces"},
    "it": {"card": "carta", "ach": "addebito diretto", "wire": "bonifico", "check": "assegno", "cash": "contanti"},
    "nl": {"card": "creditcard", "ach": "incasso", "wire": "overschrijving", "check": "cheque", "cash": "contant"},
    "codes": {"card": "CC", "ach": "DD", "wire": "BT", "check": "CK", "cash": "CS"},
}
LOCAL_NAMES = {
    "US": "USA", "GB": "Vereinigtes Königreich", "DE": "Deutschland", "FR": "France", "ES": "España", "IT": "Italia",
    "NL": "Nederland", "BE": "België", "CA": "Canada", "AU": "Australia", "IE": "Éire", "SE": "Sverige",
    "CH": "Schweiz", "AT": "Österreich", "PT": "Portugal", "PL": "Polska", "MX": "México", "BR": "Brasil",
    "JP": "Nippon", "IN": "Bharat",
}
FILE_STEMS_BY_LANG = {
    "en": ("customers", "invoices", "payments"), "de": ("kunden", "rechnungen", "zahlungen"),
    "es": ("clientes", "facturas", "pagos"), "fr": ("clients", "factures", "paiements"),
    "it": ("clienti", "fatture", "pagamenti"), "nl": ("klanten", "facturen", "betalingen"),
}
ROOTS = ["Northwind", "Contoso", "Fabrikam", "Globex", "Initech", "Umbrella", "Stark", "Wayne", "Hooli", "Acme", "Soylent", "Vandelay",
         "Cyberdyne", "Tyrell", "Oscorp", "Wonka", "Gringotts", "Sterling", "Aperture", "Weyland", "Monarch", "Nakatomi", "Bluth",
         "Zenith", "Apex", "Vertex", "Orion", "Cobalt", "Granite", "Summit", "Harbor", "Lumen", "Quanta", "Nimbus", "Pioneer",
         "Falcon", "Juniper", "Maple", "Redwood", "Beacon", "Atlas", "Meridian", "Kestrel", "Tundra"]
WORDS = ["Trading", "Logistics", "Systems", "Foods", "Labs", "Industries", "Partners", "Supply", "Energy", "Media", "Health", "Works"]
LEGAL = {"US": ["Inc", "LLC", "Corp"], "GB": ["Ltd", "PLC"], "DE": ["GmbH", "AG"], "AT": ["GmbH"], "CH": ["AG", "GmbH"],
         "FR": ["SA", "SARL"], "ES": ["SL", "SA"], "IT": ["S.p.A.", "S.r.l."], "NL": ["B.V."], "BE": ["NV"]}
LATIN1_NAMES = ["Müller & Söhne GmbH", "Café Lumière SA", "Ångström Metall AB", "Bäckerei Schröder KG", "Peña Hermanos SL", "Åkesson Fisk AB"]
UTF8_NAMES = ["Łódź Textiles Sp. z o.o.", "Şahin Lojistik A.Ş.", "Žilina Steel s.r.o."]
FIRST = ["Ana", "Ben", "Carla", "Dev", "Elif", "Finn", "Gita", "Hugo", "Iris", "Jon", "Kai", "Lena", "Milo", "Nora", "Omar", "Pia"]
LAST = ["Alder", "Brandt", "Chen", "Duarte", "Eriksen", "Fischer", "Garcia", "Haas", "Ito", "Jansen", "Kowal", "Lopez", "Moreau", "Novak"]

CUSTOMER_ID_FMT = ["C{:05d}", "CUST-{:04d}", "{:06d}", "KD{:05d}", "AC-{:05d}", "{:d}"]
INVOICE_ID_FMT = ["INV-{:06d}", "INV{:05d}", "{:07d}", "RE-2025-{:04d}", "F{:06d}", "{:d}"]
PAYMENT_ID_FMT = ["PAY-{:06d}", "P{:06d}", "{:08d}", "TXN{:07d}", "R-{:05d}", "{:d}"]
ID_START = {"customer": 1000, "invoice": 40000, "payment": 900000}
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DATE_FMTS_BY_LEVEL = {
    "easy": ["%Y-%m-%d", "%d/%m/%Y"],
    "medium": ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d.%m.%Y", "%d-%b-%Y", "%Y%m%d"],
    "hard": ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d.%m.%Y", "%d-%b-%Y", "%Y%m%d", "%b %d, %Y", "%Y-%m-%dT%H:%M:%S", "epoch", "%d-%m-%Y", "%d %b %Y"],
}
CURRENCY_SYMBOL = {"USD": "$", "EUR": "€", "GBP": "£"}

# ---------------------------------------------------------------------- specs


@dataclass
class ColSpec:
    field: str | None                 # canonical field name; None for an extra column
    header: str
    extra: str | None = None
    path: tuple[str, ...] = ()
    date_fmt: str = "%Y-%m-%d"
    pad: bool = True
    num_style: str = "plain"          # plain | us | eu | space | swiss | cents
    cur_mode: str = "column"          # amount only: column | prefix | suffix | symbol_prefix | symbol_suffix
    json_numeric: bool = False
    status_lang: str = "en"
    status_case: str = "keep"
    method_lang: str = "en"
    country_form: str = "iso2"
    phone_style: str = "e164"
    id_fmt: str = "{:d}"
    null_rate: float = 0.0
    null_token: str = ""
    opaque: bool = False

    @property
    def source(self) -> str:
        return ".".join(self.path + (self.header,))


@dataclass
class FileSpec:
    entity: str
    filename: str
    kind: str                         # csv | json | ndjson
    delimiter: str
    encoding: str
    layout: str                       # flat | nested
    wrap: str                         # list | data | records  (json only)
    cols: list[ColSpec]
    top_lines: int = 0
    footer: str = "none"              # none | total | commas
    blank_every: int = 0
    repeat_header_every: int = 0
    dup_identical: float = 0.0
    dup_conflict: float = 0.0
    quote_all: bool = False
    crlf: bool = True
    junk_amount_rate: float = 0.0
    shuffle: bool = False
    pad_ws: float = 0.0


@dataclass
class TenantSpec:
    tid: str
    difficulty: str
    seed: int
    lang: str
    currencies: list[str]
    files: dict[str, FileSpec] = field(default_factory=dict)


# ------------------------------------------------------------------ tenant plan


def _style_header(rng: random.Random, raw: str, style: str, prefix: str | None) -> str:
    toks = [t for t in re.split(r"[_\s\-]+", raw) if t]
    if prefix:
        toks = [prefix] + toks
    if style == "snake":
        return "_".join(t.lower() for t in toks)
    if style == "camel":
        return toks[0].lower() + "".join(t.capitalize() for t in toks[1:])
    if style == "pascal":
        return "".join(t.capitalize() for t in toks)
    if style == "title":
        return " ".join(t.capitalize() for t in toks)
    if style == "upper":
        return "_".join(t.upper() for t in toks)
    if style == "upper_space":
        return " ".join(t.upper() for t in toks)
    return raw


STYLES_BY_LEVEL = {
    "easy": ["snake", "title", "snake", "keep"],
    "medium": ["snake", "camel", "pascal", "title", "upper", "keep"],
    "hard": ["snake", "camel", "pascal", "title", "upper", "upper_space", "keep"],
}
OPAQUE_SCHEMES = ["COL_{:02d}", "F{:d}", "fld_{}", "Spalte{:d}", "C{:d}", "attr_{:d}"]
ENCODINGS_BY_LEVEL = {
    "easy": ["utf-8"],
    "medium": ["utf-8", "utf-8-sig", "cp1252", "latin-1"],
    "hard": ["utf-8", "utf-8-sig", "cp1252", "latin-1", "utf-8-sig"],
}


def _choose_header(rng, key, level, style, prefix, scheme, p_opaque, p_tail, idx, used, force=None):
    if force:
        raw = force
        opaque = False
    elif rng.random() < p_opaque:
        fmt = scheme
        h = fmt.format(idx + 1) if "{:" in fmt else fmt.format(chr(97 + idx % 26))
        return h, True
    else:
        tier = "tail" if rng.random() < p_tail else "core"
        raw = rng.choice(HEADERS[key][tier])
    for _ in range(8):
        h = _style_header(rng, raw, style, prefix)
        if h not in used:
            return h, False
        raw = rng.choice(HEADERS[key]["core"] + HEADERS[key]["tail"])
    return f"{raw}_{idx}", False


def build_spec(tid: str, difficulty: str, seed: int) -> TenantSpec:
    rng = random.Random(f"spec:{seed}:{tid}")
    lvl = difficulty
    lang = rng.choice({"easy": ["en"], "medium": ["en", "de", "es", "fr"], "hard": ["en", "de", "es", "fr", "it", "nl", "codes"]}[lvl])
    currencies = rng.choice([["USD"], ["EUR"], ["GBP"], ["EUR", "USD"], ["CHF"]] if lvl != "easy" else [["USD"], ["EUR"]])
    spec = TenantSpec(tid, lvl, seed, lang, currencies)

    kind_p = {"easy": (0.8, 0.2, 0.0), "medium": (0.55, 0.3, 0.15), "hard": (0.45, 0.3, 0.25)}[lvl]
    kind = rng.choices(["csv", "json", "ndjson"], weights=kind_p)[0]
    layout = "nested" if kind == "json" and lvl != "easy" and rng.random() < 0.5 else "flat"
    delimiter = rng.choice([","] if lvl == "easy" else [",", ";", "\t", "|", ";"])
    encoding = "utf-8" if kind != "csv" else rng.choice(ENCODINGS_BY_LEVEL[lvl])
    style = rng.choice(STYLES_BY_LEVEL[lvl])
    p_tail = {"easy": 0.0, "medium": 0.25, "hard": 0.5}[lvl]
    all_opaque = lvl == "hard" and rng.random() < 0.2
    p_opaque = 1.0 if all_opaque else {"easy": 0.0, "medium": 0.05, "hard": 0.3}[lvl]
    scheme = rng.choice(OPAQUE_SCHEMES)
    prefixed = lvl != "easy" and rng.random() < 0.25
    prefixes = {"customer": rng.choice(["cust", "cu", "kd"]), "invoice": rng.choice(["inv", "re", "bill"]), "payment": rng.choice(["pay", "pmt", "zlg"])}
    date_fmts = DATE_FMTS_BY_LEVEL[lvl]
    tenant_date_fmt = rng.choice(date_fmts)
    num_styles = {"easy": ["plain", "us"], "medium": ["plain", "us", "eu", "space"], "hard": ["plain", "us", "eu", "space", "swiss", "cents"]}[lvl]
    num_style = rng.choice(num_styles)
    cur_mode = rng.choices(["column", "prefix", "suffix", "symbol_prefix", "symbol_suffix"],
                           weights={"easy": [1, 0, 0, 0, 0], "medium": [7, 1, 1, 1, 1], "hard": [4, 1, 1, 1, 1]}[lvl])[0]
    if cur_mode.startswith("symbol") and any(c not in CURRENCY_SYMBOL for c in currencies):
        cur_mode = "prefix"
    if encoding == "latin-1" and cur_mode.startswith("symbol") and "EUR" in currencies:
        encoding = "cp1252"          # latin-1 has no euro sign
    country_form = rng.choice({"easy": ["iso2", "english"], "medium": ["iso2", "iso3", "english", "local"], "hard": ["iso2", "iso3", "english", "local", "local"]}[lvl])
    phone_style = rng.choice({"easy": ["e164", "spaced"], "medium": ["e164", "spaced", "dashed", "paren"], "hard": ["e164", "spaced", "dashed", "paren", "zerozero"]}[lvl])
    cust_fmt, inv_fmt, pay_fmt = (rng.choice(CUSTOMER_ID_FMT), rng.choice(INVOICE_ID_FMT), rng.choice(PAYMENT_ID_FMT))
    if lvl == "easy" and "{:d}" in (cust_fmt, inv_fmt, pay_fmt):
        cust_fmt = "C{:05d}"
    json_numeric = kind != "csv" and rng.random() < 0.6
    stems = FILE_STEMS_BY_LANG[lang if lang in FILE_STEMS_BY_LANG else "en"]
    opaque_files = lvl == "hard" and rng.random() < 0.25
    ext = {"csv": ".csv", "json": ".json", "ndjson": ".ndjson"}[kind]
    if kind == "csv" and delimiter == "\t":
        ext = rng.choice([".tsv", ".txt"])
    p_omit = {"easy": 0.05, "medium": 0.15, "hard": 0.25}[lvl]
    null_rate_hi = {"easy": 0.03, "medium": 0.06, "hard": 0.1}[lvl]

    for ei, entity in enumerate(["customer", "invoice", "payment"]):
        cols: list[ColSpec] = []
        used: set[str] = set()
        idx = 0
        for f in ENTITIES[entity]:
            if not f.required and rng.random() < p_omit:
                continue
            if entity == "invoice" and f.name == "currency" and cur_mode != "column":
                continue
            key = f"{entity}.{f.name}"
            cents_here = f.ftype == "money" and num_style == "cents"
            force = rng.choice(CENTS_HEADERS) if cents_here and rng.random() > p_opaque else None
            header, opaque = _choose_header(rng, key, lvl, style, prefixes[entity] if prefixed else None, scheme, p_opaque, p_tail, idx, used, force)
            used.add(header)
            idx += 1
            col = ColSpec(f.name, header, opaque=opaque, date_fmt=tenant_date_fmt if rng.random() < 0.8 else rng.choice(date_fmts),
                          pad=rng.random() < 0.7, num_style=num_style, cur_mode=cur_mode if f.ftype == "money" else "column",
                          json_numeric=json_numeric, status_lang=lang, status_case=rng.choice(["keep", "lower", "upper"]),
                          method_lang=lang, country_form=country_form, phone_style=phone_style)
            if f.ftype == "id":
                col.id_fmt = {"customer": cust_fmt, "invoice": inv_fmt, "payment": pay_fmt}[f.ref or entity]
            if not f.required and f.ftype not in ("money", "id"):
                col.null_rate = rng.uniform(0.0, null_rate_hi)
                col.null_token = rng.choice(["", "", "N/A", "-", "NULL"])
            cols.append(col)
        # extras: columns the product does not want
        n_extra = {"easy": rng.randint(0, 1), "medium": rng.randint(1, 3), "hard": rng.randint(2, 4)}[lvl]
        pool = [x for x in EXTRAS_BY_ENTITY[entity] if lvl != "easy" or x in EASY_EXTRAS]
        for x in rng.sample(pool, min(n_extra, len(pool))):
            h, opq = _choose_header(rng, "customer.name", lvl, style, None, scheme, p_opaque, 0, idx, used, force=rng.choice(EXTRA_HEADERS[x]))
            if rng.random() < p_opaque:
                h = scheme.format(idx + 1) if "{:" in scheme else scheme.format(chr(97 + idx % 26))
                opq = True
            used.add(h)
            idx += 1
            cols.append(ColSpec(None, h, extra=x, opaque=opq, date_fmt=tenant_date_fmt, num_style=num_style,
                                cur_mode="column", status_lang=lang))
        rng.shuffle(cols)
        if layout == "nested":
            _nest(entity, cols, rng)
        stem = stems[ei]
        fname = (f"export_{'abc'[ei]}" if opaque_files else stem) + ext
        junk_level = {"easy": 0, "medium": 1, "hard": 2}[lvl]
        spec.files[entity] = FileSpec(
            entity=entity, filename=fname, kind=kind, delimiter=delimiter, encoding=encoding, layout=layout,
            wrap=rng.choice(["list", "data", "records"]) if kind == "json" else "list", cols=cols,
            top_lines=rng.randint(0, 2) * (1 if junk_level else 0) if kind == "csv" else 0,
            footer=rng.choice(["none", "total", "commas"]) if junk_level and kind == "csv" else "none",
            blank_every=rng.choice([0, 0, 40]) if junk_level and kind == "csv" else 0,
            repeat_header_every=rng.choice([0, 30]) if junk_level == 2 and kind == "csv" else 0,
            dup_identical={"easy": rng.choice([0, 0.01]), "medium": 0.02, "hard": 0.03}[lvl],
            dup_conflict={"easy": 0.0, "medium": 0.01, "hard": 0.02}[lvl],
            quote_all=rng.random() < 0.2, crlf=rng.random() < 0.5,
            junk_amount_rate=0.015 if lvl == "hard" and entity != "customer" else 0.0,
            shuffle=rng.random() < 0.3, pad_ws=0.03 if lvl != "easy" else 0.0,
        )
    return spec


_NEST_GROUPS = {
    "customer": {"name": ["company", "firm"], "email": ["contact"], "phone": ["contact"], "country": ["address", "location"], "created_date": ["meta"]},
    "invoice": {"invoice_date": ["dates"], "due_date": ["dates"], "amount": ["totals", "amounts"], "currency": ["totals", "amounts"], "customer_id": ["customer", "billing"]},
    "payment": {"payment_date": ["details", "settlement"], "amount": ["details", "settlement"], "method": ["details", "settlement"], "invoice_id": ["applied", "invoice_ref"]},
}


def _nest(entity: str, cols: list[ColSpec], rng: random.Random) -> None:
    chosen: dict[str, str] = {}
    for c in cols:
        if c.field and c.field in _NEST_GROUPS[entity] and rng.random() < 0.85:
            options = _NEST_GROUPS[entity][c.field]
            grp = chosen.setdefault(options[0], rng.choice(options))
            c.path = (grp,)


# --------------------------------------------------------------- canonical data


def _phone(rng: random.Random) -> str:
    cc = rng.choice(["1", "44", "49", "33", "34", "31", "39", "81", "91"])
    n = "".join(rng.choice("0123456789") for _ in range(rng.choice([9, 10])))
    n = str(rng.randint(2, 9)) + n[1:]
    return "+" + cc + n


def _company(rng: random.Random, iso2: str, utf8_ok: bool, taken: set[str]) -> str:
    for _ in range(50):
        r = rng.random()
        if r < 0.08:
            name = rng.choice(LATIN1_NAMES)
        elif r < 0.11 and utf8_ok:
            name = rng.choice(UTF8_NAMES)
        elif r < 0.2:
            name = f"{rng.choice(ROOTS)}, {rng.choice(WORDS)} & Co."
        else:
            suffix = rng.choice(LEGAL.get(iso2, ["Ltd"]))
            name = f"{rng.choice(ROOTS)} {rng.choice(WORDS)} {suffix}"
        if name not in taken:
            taken.add(name)
            return name
    name = f"{rng.choice(ROOTS)} {len(taken)}"
    taken.add(name)
    return name


def make_base(spec: TenantSpec) -> dict[str, list[dict]]:
    rng = random.Random(f"base:{spec.seed}:{spec.tid}")
    utf8_ok = all(fs.encoding.startswith("utf") or fs.kind != "csv" for fs in spec.files.values())
    cfmt = _id_fmt(spec, "customer")
    ifmt = _id_fmt(spec, "invoice")
    pfmt = _id_fmt(spec, "payment")
    n_c, n_i, n_p = rng.randint(40, 80), rng.randint(80, 160), rng.randint(50, 120)
    taken: set[str] = set()
    customers = []
    cn = ID_START["customer"]
    for k in range(n_c):
        iso2 = rng.choice(list(COUNTRIES))
        name = _company(rng, iso2, utf8_ok, taken)
        slug = re.sub(r"[^a-z]", "", name.lower().replace("ü", "u").replace("é", "e"))[:12] or "co"
        cn += rng.choice([1, 1, 2, 3])
        customers.append({
            "customer_id": cfmt.format(cn), "name": name,
            "email": f"{rng.choice(['billing', 'ap', 'accounts', 'finance'])}@{slug}{k}.example",
            "phone": _phone(rng), "country": iso2,
            "created_date": (date(2019, 1, 1) + timedelta(days=rng.randint(0, 2400))).isoformat(),
        })
    invoices = []
    inn = ID_START["invoice"]
    for k in range(n_i):
        c = rng.choice(customers)
        inn += rng.choice([1, 1, 2, 3])
        d = date(2025, 1, 1) + timedelta(days=rng.randint(0, 330))
        amount = Decimal(rng.randint(5000, 2500000)) / 100
        status = rng.choices(["open", "paid", "overdue", "void", "partial"], weights=[30, 45, 12, 5, 8])[0]
        invoices.append({
            "invoice_id": ifmt.format(inn), "customer_id": c["customer_id"],
            "invoice_date": d.isoformat(), "due_date": (d + timedelta(days=rng.choice([15, 30, 45, 60]))).isoformat(),
            "currency": rng.choice(spec.currencies), "amount": f"{amount:.2f}", "status": status,
        })
    payable = [i for i in invoices if i["status"] in ("paid", "partial", "overdue")]
    payments = []
    pn = ID_START["payment"]
    for k in range(min(n_p, len(payable))):
        inv = payable[k]
        inv = rng.choice(payable) if rng.random() < 0.15 else inv
        pn += rng.choice([1, 1, 2, 3])
        d = date.fromisoformat(inv["invoice_date"]) + timedelta(days=rng.randint(1, 60))
        amt = Decimal(inv["amount"]) * (Decimal("0.5") if inv["status"] == "partial" else 1)
        payments.append({
            "payment_id": pfmt.format(pn), "invoice_id": inv["invoice_id"],
            "payment_date": d.isoformat(), "amount": f"{amt:.2f}",
            "method": rng.choice(["card", "ach", "wire", "check", "cash"]), "_cur": inv["currency"],
        })
    return {"customer": customers, "invoice": invoices, "payment": payments}


def _id_fmt(spec: TenantSpec, entity: str) -> str:
    # find the format the id column of this entity was given (the key column of that entity's file)
    for c in spec.files[entity].cols:
        if c.field == f"{entity}_id":
            return c.id_fmt
    return {"customer": "C{:05d}", "invoice": "INV-{:06d}", "payment": "PAY-{:06d}"}[entity]


# ------------------------------------------------------------------- rendering


def _fmt_date(d: date, col: ColSpec) -> str | int:
    f = col.date_fmt
    dd, mm = (f"{d.day:02d}", f"{d.month:02d}") if col.pad else (str(d.day), str(d.month))
    if f == "%Y-%m-%d":
        return d.isoformat()
    if f == "%d/%m/%Y":
        return f"{dd}/{mm}/{d.year}"
    if f == "%m/%d/%Y":
        return f"{mm}/{dd}/{d.year}"
    if f == "%d.%m.%Y":
        return f"{dd}.{mm}.{d.year}"
    if f == "%d-%m-%Y":
        return f"{dd}-{mm}-{d.year}"
    if f == "%d-%b-%Y":
        return f"{d.day:02d}-{MONTH_ABBR[d.month - 1]}-{d.year}"
    if f == "%d %b %Y":
        return f"{d.day:02d} {MONTH_ABBR[d.month - 1]} {d.year}"
    if f == "%b %d, %Y":
        return f"{MONTH_ABBR[d.month - 1]} {d.day:02d}, {d.year}"
    if f == "%Y%m%d":
        return f"{d.year}{d.month:02d}{d.day:02d}"
    if f == "%Y-%m-%dT%H:%M:%S":
        return f"{d.isoformat()}T09:30:00Z"
    if f == "epoch":
        return int(datetime(d.year, d.month, d.day, 12, tzinfo=timezone.utc).timestamp())
    raise ValueError(f)


def _group(int_part: str, sep: str) -> str:
    out = []
    while len(int_part) > 3:
        out.insert(0, int_part[-3:])
        int_part = int_part[:-3]
    out.insert(0, int_part)
    return sep.join(out)


def _fmt_number(amount: Decimal, style: str) -> str:
    sign = "-" if amount < 0 else ""
    whole, frac = f"{abs(amount):.2f}".split(".")
    if style == "plain":
        return f"{sign}{whole}.{frac}"
    if style == "us":
        return f"{sign}{_group(whole, ',')}.{frac}"
    if style == "eu":
        return f"{sign}{_group(whole, '.')},{frac}"
    if style == "space":
        return f"{sign}{_group(whole, ' ')},{frac}"
    if style == "swiss":
        return f"{sign}{_group(whole, chr(39))}.{frac}"
    raise ValueError(style)


def _fmt_money(amount: Decimal, cur: str, col: ColSpec, is_json: bool):
    if col.num_style == "cents":
        return int(amount * 100)
    if is_json and col.json_numeric and col.cur_mode == "column" and col.num_style == "plain":
        return float(amount)
    num = _fmt_number(amount, col.num_style)
    mode = col.cur_mode
    if mode == "prefix":
        return f"{cur} {num}"
    if mode == "suffix":
        return f"{num} {cur}"
    if mode == "symbol_prefix":
        return f"{CURRENCY_SYMBOL[cur]}{num}"
    if mode == "symbol_suffix":
        return f"{num} {CURRENCY_SYMBOL[cur]}"
    return num


def _fmt_phone(p: str, style: str) -> str:
    digits = p[1:]
    for cc_len in (1, 2):
        cc, n = digits[:cc_len], digits[cc_len:]
        if cc in {"1", "44", "49", "33", "34", "31", "39", "81", "91"} and len(n) >= 9:
            break
    a, b, c = n[:3], n[3:6], n[6:]
    return {"e164": p, "spaced": f"+{cc} {a} {b} {c}", "dashed": f"+{cc}-{a}-{b}-{c}",
            "paren": f"+{cc} ({a}) {b}-{c}", "zerozero": f"00{cc} {a} {b} {c}"}[style]


def _case(s: str, how: str) -> str:
    return s.lower() if how == "lower" else s.upper() if how == "upper" else s


def _extra_value(kind: str, rec: dict, rng: random.Random, col: ColSpec, entity: str, is_json: bool):
    if kind == "sales_rep":
        return f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    if kind == "region":
        return f"R{rng.randint(1, 9)}"
    if kind == "segment":
        return rng.choice(["smb", "mid", "enterprise"])
    if kind == "legacy_flag":
        return rng.choice(["Y", "N"])
    if kind == "updated_at":
        base = rec.get("invoice_date") or rec.get("payment_date") or rec.get("created_date") or "2025-06-01"
        d = date.fromisoformat(base) + timedelta(days=rng.randint(0, 120))
        return _fmt_date(d, col)
    if kind == "notes":
        return rng.choice(["call back, after 5pm", "disputed; see ticket", "", "ok", "net 30 agreed, per contract", "VIP"])
    if kind == "tax_amount":
        return _fmt_money((Decimal(rec["amount"]) * Decimal("0.2")).quantize(Decimal("0.01")), rec["currency"], col, is_json)
    if kind == "balance":
        bal = Decimal(rec["amount"]) if rec["status"] in ("open", "overdue") else Decimal("0")
        return _fmt_money(bal, rec["currency"], col, is_json)
    if kind == "po_number":
        return f"PO-{rng.randint(10000, 99999)}"
    if kind == "terms":
        return rng.choice(["Net 15", "Net 30", "Net 45", "Net 60"])
    if kind == "bank_ref":
        return f"BK{rng.randint(1000000, 9999999)}"
    if kind == "batch_id":
        return f"B{rng.randint(100, 140)}"
    if kind == "currency_col":
        return rec.get("_cur", "EUR")
    if kind == "row_hash":
        return "%08x" % rng.getrandbits(32)
    raise ValueError(kind)


def _cell(col: ColSpec, rec: dict, rng: random.Random, is_json: bool, canon: dict, junk: bool):
    f = col.field
    if f is None:
        return _extra_value(col.extra, rec, rng, col, "", is_json)
    v = rec.get(f)
    nullable = col.null_rate > 0
    if nullable and rng.random() < col.null_rate:
        canon[f] = None
        return None if is_json and col.null_token == "" and rng.random() < 0.5 else col.null_token
    if f in ("created_date", "invoice_date", "due_date", "payment_date"):
        return _fmt_date(date.fromisoformat(v), col)
    if f == "amount":
        if junk:
            return rng.choice(["TBD", "see notes", "pending"])
        cur = rec.get("currency") or rec.get("_cur") or "EUR"
        return _fmt_money(Decimal(v), cur, col, is_json)
    if f == "currency":
        return v
    if f == "status":
        return _case(STATUS_WORDS[col.status_lang][v], col.status_case)
    if f == "method":
        return _case(METHOD_WORDS[col.method_lang][v], col.status_case)
    if f == "country":
        iso3, en = COUNTRIES[v]
        return {"iso2": v, "iso3": iso3, "english": en, "local": LOCAL_NAMES[v]}[col.country_form]
    if f == "phone":
        return _fmt_phone(v, col.phone_style)
    if f == "email":
        r = rng.random()
        return v.upper() if r < 0.05 else v.capitalize() if r < 0.1 else v
    return v


def render_file(fs: FileSpec, records: list[dict], seed_key: str) -> tuple[bytes, list[dict], list[str]]:
    """Return (file bytes, canonical records that should load, keys that should be quarantined)."""
    rng = random.Random(f"render:{seed_key}:{fs.filename}")
    is_json = fs.kind != "csv"
    key_field = f"{fs.entity}_id"
    present = {c.field for c in fs.cols if c.field}
    # emit order with duplicates
    emitted: list[tuple[dict, bool]] = []                 # (record, is_conflict_copy)
    for r in records:
        emitted.append((dict(r), False))
    if fs.shuffle:
        rng.shuffle(emitted)
    out_rows: list[tuple[dict, bool]] = []
    for r, _ in emitted:
        out_rows.append((r, False))
        x = rng.random()
        if x < fs.dup_identical:
            out_rows.append((dict(r), False))
        elif x < fs.dup_identical + fs.dup_conflict:
            changed = dict(r)
            if fs.entity == "customer":
                changed["name"] = r["name"] + " (renamed)"
            else:
                changed["amount"] = f"{Decimal(r['amount']) + Decimal('1.00'):.2f}"
            out_rows.append((changed, True))
    canon_by_key: dict[str, dict] = {}
    quarantined: list[str] = []
    cell_rows: list[dict] = []
    for r, _ in out_rows:
        junk = fs.junk_amount_rate > 0 and "amount" in present and rng.random() < fs.junk_amount_rate
        canon = {f: (r.get(f) if f in present else None) for f in [x.name for x in _canon_fields(fs.entity)]}
        if fs.entity == "invoice" and "currency" not in present:
            canon["currency"] = r["currency"]
        cells = {}
        for c in fs.cols:
            v = _cell(c, r, rng, is_json, canon, junk)
            if isinstance(v, str) and fs.pad_ws and rng.random() < fs.pad_ws:
                v = f" {v} "
            cells[c.source] = (c, v)
        cell_rows.append({"cells": cells, "canon": canon, "junk": junk, "key": r[key_field]})
        k = r[key_field]
        if junk:
            quarantined.append(k)
        else:
            canon_by_key[k] = canon
    quarantined = [k for k in quarantined if k not in canon_by_key]
    data = _serialize(fs, cell_rows, rng)
    return data, list(canon_by_key.values()), quarantined


def _canon_fields(entity: str):
    return ENTITIES[entity]


def _serialize(fs: FileSpec, cell_rows: list[dict], rng: random.Random) -> bytes:
    headers = [c.source for c in fs.cols]
    if fs.kind == "csv":
        buf = io.StringIO()
        nl = "\r\n" if fs.crlf else "\n"
        for i in range(fs.top_lines):
            buf.write(["Export from LegacyERP v4.2", "Generated: 2025-11-03 08:12", ""][i % 3] + nl)
        w = csv.writer(buf, delimiter=fs.delimiter, quoting=csv.QUOTE_ALL if fs.quote_all else csv.QUOTE_MINIMAL, lineterminator=nl)
        w.writerow(headers)
        for i, row in enumerate(cell_rows, 1):
            w.writerow(["" if row["cells"][h][1] is None else row["cells"][h][1] for h in headers])
            if fs.repeat_header_every and i % fs.repeat_header_every == 0:
                w.writerow(headers)
            if fs.blank_every and i % fs.blank_every == 0:
                buf.write(nl)
        if fs.footer == "total":
            buf.write(f"Total records: {len(cell_rows)}{nl}")
        elif fs.footer == "commas":
            buf.write(fs.delimiter * (len(headers) - 1) + nl)
        text = buf.getvalue()
        enc = fs.encoding
        data = text.encode(enc)
        return data
    objs = []
    for row in cell_rows:
        obj: dict = {}
        for h in headers:
            c, v = row["cells"][h]
            node = obj
            for p in c.path:
                node = node.setdefault(p, {})
            node[c.header] = v
        objs.append(obj)
    if fs.kind == "ndjson":
        return ("\n".join(json.dumps(o, ensure_ascii=False) for o in objs) + "\n").encode("utf-8")
    payload = objs if fs.wrap == "list" else {"meta": {"exported": "2025-11-03", "count": len(objs)}, fs.wrap: objs}
    return json.dumps(payload, ensure_ascii=False, indent=1 if rng.random() < 0.5 else None).encode("utf-8")


# ------------------------------------------------------------------ truth + IO


def _truth_columns(fs: FileSpec) -> dict[str, str | None]:
    return {c.source: (f"{fs.entity}.{c.field}" if c.field else None) for c in fs.cols}


def _write_sync(root: Path, folder: str, spec: TenantSpec, base: dict[str, list[dict]], tag: str) -> dict:
    out = root / folder
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    truth: dict = {"files": {}, "records": {}, "expected_quarantine": {}}
    for entity, fs in spec.files.items():
        data, canon, quarantined = render_file(fs, base[entity], f"{spec.tid}:{tag}")
        (out / fs.filename).write_bytes(data)
        truth["files"][fs.filename] = {"entity": entity, "columns": _truth_columns(fs), "kind": fs.kind,
                                       "encoding": fs.encoding, "delimiter": fs.delimiter, "layout": fs.layout}
        truth["records"][entity] = canon
        truth["expected_quarantine"][entity] = quarantined
    return truth


@dataclass
class Tenant:
    tid: str
    difficulty: str
    root: Path

    def data_dir(self, sync: str = "base") -> Path:
        return self.root / sync

    def files(self, sync: str = "base") -> list[Path]:
        return sorted(p for p in self.data_dir(sync).iterdir() if p.is_file())

    def truth(self, sync: str = "base") -> dict:
        name = "truth.json" if sync == "base" else f"truth_{sync}.json"
        return json.loads((self.root / name).read_text(encoding="utf-8"))


def load_tenants(out_dir: Path) -> list[Tenant]:
    out = []
    for d in sorted(Path(out_dir).iterdir()):
        if d.is_dir() and (d / "meta.json").exists():             # difficulty lives here so listing tenants never opens a truth file
            out.append(Tenant(d.name, json.loads((d / "meta.json").read_text(encoding="utf-8"))["difficulty"], d))
    return out


def generate_all(out_dir: Path, n: int = N_TENANTS, seed: int = GEN_SEED, plan: list[str] | None = None) -> list[Tenant]:
    from .drift_inject import make_syncs

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = plan or [DIFFICULTY_PLAN[i % len(DIFFICULTY_PLAN)] for i in range(n)]
    tenants = []
    for i in range(n):
        tid = f"t{i:02d}"
        spec = build_spec(tid, plan[i], seed)
        base = make_base(spec)
        root = out_dir / tid
        root.mkdir(parents=True, exist_ok=True)
        truth = _write_sync(root, "base", spec, base, "base")
        truth.update({"tenant_id": tid, "difficulty": spec.difficulty, "seed": seed, "lang": spec.lang})
        (root / "truth.json").write_text(json.dumps(truth, indent=1, ensure_ascii=False), encoding="utf-8")
        (root / "meta.json").write_text(json.dumps({"tenant_id": tid, "difficulty": spec.difficulty, "lang": spec.lang, "seed": seed}), encoding="utf-8")
        make_syncs(root, spec, base, seed)
        tenants.append(Tenant(tid, spec.difficulty, root))
    return tenants


def lexicon_coverage(tenants: list[Tenant]) -> dict:
    """Share of mapped source headers whose tokens the mapper's lexicon has never seen."""
    from .schema import LEXICON, header_tokens

    known = {t for phrases in LEXICON.values() for p, _ in phrases for t in p}
    total = unseen = 0
    for t in tenants:
        for meta in t.truth()["files"].values():
            for src, target in meta["columns"].items():
                if target is None:
                    continue
                toks = header_tokens(src.split(".")[-1])
                total += 1
                if toks and not any(tok in known for tok in toks):
                    unseen += 1
    return {"columns": total, "no_known_token": unseen, "share": round(unseen / max(total, 1), 3)}
