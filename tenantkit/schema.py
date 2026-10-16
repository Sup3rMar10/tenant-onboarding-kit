"""The canonical schema, and the vocabularies the mapper is allowed to know.

Everything in this file is what a practitioner would write down before meeting a
new customer: the target fields, the common ways people name them in a handful
of languages, and the words used for statuses, payment methods and countries.
The synthetic generator has its own, larger vocabulary (see generate.py), so
that some of what it produces is outside what is written here.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class Field:
    name: str
    ftype: str            # id, text, email, phone, country, date, money, currency, status, method
    required: bool = False
    key: bool = False
    ref: str | None = None  # entity this id points at, for foreign keys


ENTITIES: dict[str, tuple[Field, ...]] = {
    "customer": (
        Field("customer_id", "id", required=True, key=True),
        Field("name", "text", required=True),
        Field("email", "email"),
        Field("phone", "phone"),
        Field("country", "country"),
        Field("created_date", "date"),
    ),
    "invoice": (
        Field("invoice_id", "id", required=True, key=True),
        Field("customer_id", "id", required=True, ref="customer"),
        Field("invoice_date", "date", required=True),
        Field("due_date", "date"),
        Field("currency", "currency", required=True),
        Field("amount", "money", required=True),
        Field("status", "status"),
    ),
    "payment": (
        Field("payment_id", "id", required=True, key=True),
        Field("invoice_id", "id", required=True, ref="invoice"),
        Field("payment_date", "date", required=True),
        Field("amount", "money", required=True),
        Field("method", "method"),
    ),
}

FILE_STEMS = {"customer": "customers", "invoice": "invoices", "payment": "payments"}


def field_names(entity: str) -> list[str]:
    return [f.name for f in ENTITIES[entity]]


def get_field(entity: str, name: str) -> Field:
    for f in ENTITIES[entity]:
        if f.name == name:
            return f
    raise KeyError(f"{entity}.{name}")


def all_targets() -> list[tuple[str, str]]:
    return [(e, f.name) for e, fs in ENTITIES.items() for f in fs]


# ---------------------------------------------------------------- text helpers

_CAMEL = re.compile(r"([a-z0-9])([A-Z])")
_NONALNUM = re.compile(r"[^a-z0-9]+")


def strip_accents(s: str) -> str:
    s = s.replace("ß", "ss")
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def norm_text(s: str) -> str:
    return strip_accents(s).lower().strip()


def header_tokens(header: str) -> tuple[str, ...]:
    """'InvoiceDate', 'invoice_date', 'Invoice Date', 'invoice.date' -> ('invoice', 'date')."""
    s = _CAMEL.sub(r"\1 \2", header)
    s = _NONALNUM.sub(" ", norm_text(s))
    return tuple(t for t in s.split() if t)


# ----------------------------------------------------------- header vocabulary
# A leading "~" marks a generic phrase ("id", "date") that is weak evidence.

_RAW_LEXICON = {
    "customer.customer_id": "customer id|customer number|customer no|customer nr|customer code|customer ref|customer|cust id|cust no|cust nr|cust num|cust|client id|client number|client no|client code|account id|account number|account no|kundennummer|kundennr|kunden nr|kunden id|kunde nr|id cliente|numero cliente|codigo cliente|~id|~code|~number|~no|~ref|~key",
    "customer.name": "name|customer name|company name|company|client name|account name|organization|organisation|business name|legal name|firma|firmenname|kundenname|nombre|razon social|nom|raison sociale|~customer|~client",
    "customer.email": "email|e mail|email address|mail|contact email|billing email|ar email|correo|e mail adresse|courriel",
    "customer.phone": "phone|telephone|tel|phone number|phone no|mobile|contact phone|telefon|telefono|telephone number",
    "customer.country": "country|country code|nation|land|pais|pays|cntry|ctry|iso country",
    "customer.created_date": "created|created date|created at|creation date|customer since|since|signup date|registered|registration date|onboarded|erstellt|erstellungsdatum|anlagedatum|fecha alta|fecha creacion|date creation|~date",
    "invoice.invoice_id": "invoice id|invoice number|invoice no|invoice nr|invoice num|invoice|inv no|inv nr|inv id|inv num|inv|bill number|bill no|bill id|document number|document no|doc no|rechnungsnummer|rechnungsnr|rechnung nr|numero factura|factura|no factura|numero de facture|~id|~number|~no|~ref|~document",
    "invoice.customer_id": "customer id|customer number|customer no|customer|cust id|cust no|cust|client id|client|client no|bill to|account id|account|kundennummer|kundennr|kunde|id cliente|cliente|~id|~ref",
    "invoice.invoice_date": "invoice date|inv date|date issued|issue date|issued|issued on|billing date|bill date|document date|doc date|posting date|rechnungsdatum|belegdatum|fecha factura|fecha emision|date facture|~date|~created",
    "invoice.due_date": "due date|due|due on|payment due|payable by|pay by|net due|maturity|maturity date|due by|falligkeitsdatum|faelligkeitsdatum|fallig|falligkeit|fecha vencimiento|vencimiento|echeance|date echeance",
    "invoice.currency": "currency|currency code|curr|ccy|cur|waehrung|wahrung|moneda|devise|iso currency",
    "invoice.amount": "amount|total|total amount|invoice amount|invoice total|gross amount|gross|grand total|amount due|balance due|sum|value|betrag|rechnungsbetrag|gesamtbetrag|summe|importe|monto|total factura|montant|amt|total due|~balance|~net",
    "invoice.status": "status|invoice status|state|payment status|zahlungsstatus|estado|statut|~flag",
    "payment.payment_id": "payment id|payment number|payment no|payment ref|payment reference|receipt number|receipt no|receipt id|remittance id|transaction id|txn id|trx id|transaction number|zahlungsnummer|zahlungs id|id pago|numero pago|pay id|pmt id|pmt no|~id|~number|~ref",
    "payment.invoice_id": "invoice id|invoice number|invoice no|invoice|inv no|inv id|applied to|applied to invoice|against invoice|rechnungsnummer|rechnungsnr|factura|numero factura|~ref|~id",
    "payment.payment_date": "payment date|paid on|paid date|date paid|received|received on|received date|receipt date|value date|settlement date|transaction date|txn date|zahlungsdatum|eingangsdatum|fecha pago|fecha de pago|date paiement|~date",
    "payment.amount": "amount|payment amount|paid amount|amount paid|paid|received amount|total|sum|value|betrag|zahlbetrag|zahlungsbetrag|importe|monto|importe pagado|montant|amt|pmt amt|~net",
    "payment.method": "method|payment method|pay method|payment type|type|mode|tender|channel|zahlungsart|zahlungsmethode|metodo|metodo de pago|forma de pago|mode de paiement|pmt method|pmt type|~via",
}

GENERIC_WEIGHT = 0.55


def _compile_lexicon() -> dict[str, list[tuple[tuple[str, ...], float]]]:
    out = {}
    for key, raw in _RAW_LEXICON.items():
        phrases = []
        for p in raw.split("|"):
            weight = 1.0
            if p.startswith("~"):
                weight, p = GENERIC_WEIGHT, p[1:]
            toks = header_tokens(p)
            if toks:
                phrases.append((toks, weight))
        out[key] = phrases
    return out


LEXICON = _compile_lexicon()

# Words that mark a column as bookkeeping rather than business data (audit stamps, hashes, free text,
# derived amounts). A name containing one of these makes a match to a real field less likely.
IGNORE_TOKENS = {"updated", "update", "modified", "modify", "last", "changed", "audit", "hash", "checksum", "etag", "batch",
                 "note", "notes", "comment", "comments", "remarks", "memo", "legacy", "migrated", "tax", "vat", "po"}

# What a file called customers.csv / Rechnungen.csv is about.
ENTITY_WORDS = {
    "customer": {"customer", "customers", "client", "clients", "account", "accounts", "kunde", "kunden", "cliente", "clientes"},
    "invoice": {"invoice", "invoices", "bill", "bills", "rechnung", "rechnungen", "factura", "facturas", "facture", "factures", "ar"},
    "payment": {"payment", "payments", "receipt", "receipts", "remittance", "remittances", "zahlung", "zahlungen", "pago", "pagos", "paiement", "paiements"},
}

# ------------------------------------------------------------- value vocabulary

STATUS_VOCAB = {
    "open": {"open", "unpaid", "pending", "outstanding", "offen", "pendiente", "ouvert", "o"},
    "paid": {"paid", "settled", "closed", "bezahlt", "pagada", "pagado", "paye", "p"},
    "overdue": {"overdue", "late", "past due", "uberfallig", "ueberfaellig", "vencida", "en retard", "od"},
    "void": {"void", "cancelled", "canceled", "storniert", "anulada", "annule", "v"},
    "partial": {"partial", "partially paid", "teilbezahlt", "parcial", "pp"},
}
STATUS_LOOKUP = {norm_text(w): canon for canon, ws in STATUS_VOCAB.items() for w in ws}

METHOD_VOCAB = {
    "card": {"card", "credit card", "cc", "visa", "mastercard", "kreditkarte", "tarjeta", "carte"},
    "ach": {"ach", "direct debit", "sepa", "lastschrift", "debito", "prelevement"},
    "wire": {"wire", "bank transfer", "transfer", "uberweisung", "transferencia", "virement", "wire transfer"},
    "check": {"check", "cheque", "scheck"},
    "cash": {"cash", "bargeld", "efectivo", "especes"},
}
METHOD_LOOKUP = {norm_text(w): canon for canon, ws in METHOD_VOCAB.items() for w in ws}

# iso2 -> (iso3, English name)
COUNTRIES = {
    "US": ("USA", "United States"), "GB": ("GBR", "United Kingdom"), "DE": ("DEU", "Germany"),
    "FR": ("FRA", "France"), "ES": ("ESP", "Spain"), "IT": ("ITA", "Italy"),
    "NL": ("NLD", "Netherlands"), "BE": ("BEL", "Belgium"), "CA": ("CAN", "Canada"),
    "AU": ("AUS", "Australia"), "IE": ("IRL", "Ireland"), "SE": ("SWE", "Sweden"),
    "CH": ("CHE", "Switzerland"), "AT": ("AUT", "Austria"), "PT": ("PRT", "Portugal"),
    "PL": ("POL", "Poland"), "MX": ("MEX", "Mexico"), "BR": ("BRA", "Brazil"),
    "JP": ("JPN", "Japan"), "IN": ("IND", "India"),
}
# the native-language names the mapper happens to know. The generator uses more.
KNOWN_LOCAL_NAMES = {"DE": "Deutschland", "ES": "Espana", "AT": "Osterreich", "CH": "Schweiz", "US": "USA"}


def _country_lookup() -> dict[str, str]:
    out: dict[str, str] = {}
    for iso2, (iso3, en) in COUNTRIES.items():
        for w in (iso2, iso3, en):
            out[norm_text(w)] = iso2
    for iso2, local in KNOWN_LOCAL_NAMES.items():
        out[norm_text(local)] = iso2
    out.update({"uk": "GB", "u.s.": "US", "united states of america": "US", "great britain": "GB"})
    return out


COUNTRY_LOOKUP = _country_lookup()

CURRENCY_CODES = {"USD", "EUR", "GBP", "CAD", "AUD", "CHF", "JPY", "SEK", "PLN", "BRL", "MXN", "INR"}
CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP"}

COMPANY_SUFFIXES = {"inc", "llc", "ltd", "gmbh", "ag", "sa", "bv", "corp", "co", "plc", "srl", "sarl", "kg", "oy", "ab", "spa", "company", "group", "holdings", "partners"}

NULL_TOKENS = {"", "n/a", "na", "null", "none", "-", "--", "nan", "nil", "#n/a", "(blank)", "?"}
