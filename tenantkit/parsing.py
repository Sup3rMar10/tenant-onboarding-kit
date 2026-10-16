"""Parsing and formatting-independent value handling: dates, money, phones, ids.

Written without strptime so it is fast on a few thousand cells and does not
depend on the process locale.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation

from .schema import CURRENCY_CODES, CURRENCY_SYMBOLS, NULL_TOKENS, norm_text

MONTHS = {m: i + 1 for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split())}

# name -> (regex, group order)
_DATE_PATTERNS: dict[str, tuple[re.Pattern, tuple[str, ...]]] = {
    "%Y-%m-%d": (re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$"), ("y", "m", "d")),
    "%Y/%m/%d": (re.compile(r"^(\d{4})/(\d{1,2})/(\d{1,2})$"), ("y", "m", "d")),
    "%d/%m/%Y": (re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$"), ("d", "m", "y")),
    "%m/%d/%Y": (re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$"), ("m", "d", "y")),
    "%d.%m.%Y": (re.compile(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$"), ("d", "m", "y")),
    "%d-%m-%Y": (re.compile(r"^(\d{1,2})-(\d{1,2})-(\d{4})$"), ("d", "m", "y")),
    "%d-%b-%Y": (re.compile(r"^(\d{1,2})-([A-Za-z]{3})-(\d{4})$"), ("d", "b", "y")),
    "%d %b %Y": (re.compile(r"^(\d{1,2}) ([A-Za-z]{3}) (\d{4})$"), ("d", "b", "y")),
    "%b %d, %Y": (re.compile(r"^([A-Za-z]{3}) (\d{1,2}), (\d{4})$"), ("b", "d", "y")),
    "%Y%m%d": (re.compile(r"^(\d{4})(\d{2})(\d{2})$"), ("y", "m", "d")),
    "%Y-%m-%dT%H:%M:%S": (re.compile(r"^(\d{4})-(\d{2})-(\d{2})[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?$"), ("y", "m", "d")),
}
DATE_FORMATS = list(_DATE_PATTERNS) + ["epoch"]


def parse_date(s: str, fmt: str) -> date | None:
    s = s.strip()
    if fmt == "epoch":
        if re.fullmatch(r"\d{10}", s):
            try:
                d = datetime.fromtimestamp(int(s), tz=timezone.utc).date()
            except (OverflowError, OSError, ValueError):
                return None
            return d if 1990 <= d.year <= 2100 else None
        return None
    pat = _DATE_PATTERNS.get(fmt)
    if pat is None:
        return None
    m = pat[0].match(s)
    if not m:
        return None
    parts = dict(zip(pat[1], m.groups()))
    try:
        month = MONTHS[parts["b"].lower()] if "b" in parts else int(parts["m"])
        d = date(int(parts["y"]), month, int(parts["d"]))
    except (KeyError, ValueError):
        return None
    return d if 1970 <= d.year <= 2100 else None     # 00900001 is an id, not a date


def format_hint(fmt: str) -> str:
    return fmt


def is_null(s: str) -> bool:
    return norm_text(s) in NULL_TOKENS


# ------------------------------------------------------------------ money

_SPACE_THOUSANDS = re.compile(r"(?<=\d)[\s  '](?=\d{3}(?:\D|$))")
_CORE = re.compile(r"^-?[\d.,]*\d[\d.,]*-?$")


def split_money(raw: str) -> tuple[str | None, str | None, bool]:
    """Return (digits-and-separators, currency, negative) or (None, None, False)."""
    s = raw.strip().replace(" ", " ")
    if not s:
        return None, None, False
    currency = None
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative, s = True, s[1:-1].strip()
    m = re.match(r"^([A-Za-z]{3})\s*(.*)$", s)
    if m and m.group(1).upper() in CURRENCY_CODES:
        currency, s = m.group(1).upper(), m.group(2)
    m = re.match(r"^(.*?)\s*([A-Za-z]{3})$", s)
    if m and currency is None and m.group(2).upper() in CURRENCY_CODES:
        currency, s = m.group(2).upper(), m.group(1)
    for sym, code in CURRENCY_SYMBOLS.items():
        if s.startswith(sym):
            currency, s = code, s[len(sym):].strip()
        elif s.endswith(sym):
            currency, s = code, s[: -len(sym)].strip()
    s = _SPACE_THOUSANDS.sub("", s.strip())
    if s.endswith("-") and not s.startswith("-"):
        negative, s = True, s[:-1]
    if s.startswith("-"):
        negative, s = True, s[1:]
    if not _CORE.match(s):
        return None, currency, False
    return s, currency, negative


def decimal_vote(core: str) -> str | None:
    """Which character is the decimal separator in this one number, if it can be told."""
    has_dot, has_comma = "." in core, "," in core
    if has_dot and has_comma:
        return "." if core.rfind(".") > core.rfind(",") else ","
    sep = "." if has_dot else "," if has_comma else None
    if sep is None:
        return None
    if core.count(sep) > 1:
        return "," if sep == "." else "."          # repeated: thousands separator
    after = len(core) - core.rfind(sep) - 1
    return None if after == 3 else sep              # "1,234" could be either: no vote


def infer_decimal(cores: list[str]) -> str:
    votes = {".": 0, ",": 0}
    for c in cores:
        v = decimal_vote(c)
        if v:
            votes[v] += 1
    if votes[","] > votes["."]:
        return ","
    return "."


def parse_core(core: str, decimal: str) -> Decimal | None:
    thousands = "," if decimal == "." else "."
    c = core.replace(thousands, "")
    if decimal == ",":
        c = c.replace(",", ".")
    if c.count(".") > 1:
        return None
    try:
        return Decimal(c)
    except InvalidOperation:
        return None


def parse_money(raw: str, decimal: str = ".", scale: Decimal = Decimal(1)) -> tuple[Decimal | None, str | None]:
    core, currency, neg = split_money(raw)
    if core is None:
        return None, currency
    val = parse_core(core, decimal)
    if val is None:
        return None, currency
    val = val * scale
    if neg:
        val = -val
    return val.quantize(Decimal("0.01")), currency


# ------------------------------------------------------------------ other

_EMAIL = re.compile(r"^[A-Za-z0-9._%+\-']+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


def is_email(s: str) -> bool:
    return bool(_EMAIL.match(s.strip()))


def normalize_phone(s: str) -> str | None:
    """'+1 (415) 555-0123', '001 415 555 0123' -> '+14155550123'. None if it cannot be made international."""
    t = s.strip()
    plus = t.startswith("+") or t.startswith("00")
    digits = re.sub(r"\D", "", t)
    if t.startswith("00"):
        digits = digits[2:]
    if not plus or not 7 <= len(digits) <= 15:
        return None
    return "+" + digits


_PHONE_LIKE = re.compile(r"^\+?[\d\s().\-]{7,22}$")


def looks_like_phone(s: str) -> bool:
    t = s.strip()
    if not _PHONE_LIKE.match(t):
        return False
    digits = re.sub(r"\D", "", t)
    if not 7 <= len(digits) <= 15:
        return False
    if re.search(r"[\s().\-]", t):
        return True
    if t.startswith("+"):
        return len(digits) >= 8
    return t.startswith("00") and len(digits) >= 11      # 0040012 is a zero-padded id, not a phone number


def shape(s: str) -> str:
    """'C-00123' -> 'A-9' style signature: runs of digits, upper, lower collapsed."""
    out = []
    for ch in s:
        c = "9" if ch.isdigit() else "A" if ch.isupper() else "a" if ch.islower() else ch
        if not out or out[-1] != c:
            out.append(c)
    return "".join(out)


def id_shaped(s: str) -> bool:
    """Short, no spaces, contains a digit run: INV-2025-0042, C00123, 100234."""
    t = s.strip()
    return 2 <= len(t) <= 24 and " " not in t and bool(re.search(r"\d{2,}", t)) and bool(re.fullmatch(r"[A-Za-z0-9_/\-#]+", t))
