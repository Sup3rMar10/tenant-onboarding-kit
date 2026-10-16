"""Value profiling: what do the cells of this column look like?

A profile is the evidence the mapper and the drift detector work from. It holds
fractions (how many cells look like an email, a date, an amount, a known status word),
shape and cardinality statistics, and a small hashed sketch of the values so two
columns can be compared for overlap without storing the values themselves.
"""
from __future__ import annotations

import hashlib
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal

from .parsing import (DATE_FORMATS, id_shaped, infer_decimal, is_email, is_null, looks_like_phone, parse_core, parse_date, shape,
                      split_money)
from .readers import RawTable
from .schema import (COMPANY_SUFFIXES, COUNTRY_LOOKUP, CURRENCY_CODES, CURRENCY_SYMBOLS, METHOD_LOOKUP, STATUS_LOOKUP, norm_text)

SKETCH_K = 64
_NUMERIC_KINDS = {"money", "int", "id"}


@dataclass
class Profile:
    name: str
    n: int = 0
    n_nonempty: int = 0
    n_distinct: int = 0
    null_rate: float = 1.0
    uniq_ratio: float = 0.0
    email: float = 0.0
    phone: float = 0.0
    date: float = 0.0
    date_format: str | None = None
    date_ambiguous: bool = False
    date_candidates: list[str] = field(default_factory=list)   # every format tied for best (d/m vs m/d)
    money: float = 0.0
    money_decimals: float = 0.0
    decimal: str = "."
    currency_token: float = 0.0
    integer: float = 0.0
    idshape: float = 0.0
    len_mean: float = 0.0
    len_cv: float = 0.0
    currency: float = 0.0
    status: float = 0.0
    method: float = 0.0
    country: float = 0.0
    alpha: float = 0.0
    company: float = 0.0
    words_mean: float = 0.0
    shape_top: str = ""
    shape_share: float = 0.0
    num_min: float | None = None
    num_max: float | None = None
    num_mean: float | None = None
    kind: str = "empty"
    sketch: list[int] = field(default_factory=list)
    values: frozenset = frozenset()          # distinct stripped values, for inclusion checks; never serialized

    def fingerprint(self) -> dict:
        """What the config remembers about this column. Hashed sketch, no raw values."""
        return {"kind": self.kind, "date_format": self.date_format, "date_candidates": self.date_candidates, "decimal": self.decimal if self.kind == "money" else None,
                "shape_top": self.shape_top, "shape_share": round(self.shape_share, 3), "uniq_ratio": round(self.uniq_ratio, 3),
                "null_rate": round(self.null_rate, 3), "len_mean": round(self.len_mean, 2),
                "money_decimals": round(self.money_decimals, 3),
                "num_min": self.num_min, "num_max": self.num_max, "num_mean": self.num_mean, "sketch": self.sketch}


def _hash(v: str) -> int:
    return int.from_bytes(hashlib.sha1(v.encode("utf-8")).digest()[:6], "big")


def _classify(p: Profile) -> str:
    if p.n_nonempty == 0:
        return "empty"
    if p.email >= 0.8:
        return "email"
    if p.date >= 0.8:
        return "date"
    if p.currency >= 0.8:
        return "currency"
    if p.status >= 0.7:
        return "status"
    if p.method >= 0.7:
        return "method"
    if p.country >= 0.8:
        return "country"
    if p.phone >= 0.7:
        return "phone"
    if p.money >= 0.85:
        if p.money_decimals >= 0.3 or p.currency_token >= 0.3:
            return "money"
        if p.integer >= 0.95:
            return "id" if p.len_cv <= 0.1 else "int"
        return "money"
    if p.idshape >= 0.8:
        return "id"
    return "text"


def profile_column(name: str, cells: list[str]) -> Profile:
    p = Profile(name=name, n=len(cells))
    vals = [c.strip() for c in cells if not is_null(c)]
    p.n_nonempty = len(vals)
    p.null_rate = 1 - len(vals) / len(cells) if cells else 1.0
    if not vals:
        return p
    distinct = set(vals)
    p.values = frozenset(distinct)
    p.n_distinct = len(distinct)
    p.uniq_ratio = len(distinct) / len(vals)
    n = len(vals)
    p.email = sum(is_email(v) for v in vals) / n
    p.phone = sum(looks_like_phone(v) for v in vals) / n

    counts = {fmt: sum(parse_date(v, fmt) is not None for v in vals) for fmt in DATE_FORMATS}
    best_n = max(counts.values())
    best = [f for f, k in counts.items() if k == best_n and k > 0]
    p.date = best_n / n
    p.date_format = best[0] if best else None
    p.date_ambiguous = len(best) > 1
    p.date_candidates = best

    cores, cur_tokens, nums = [], 0, []
    for v in vals:
        core, cur, _ = split_money(v)
        if core is not None:
            cores.append(core)
        if cur:
            cur_tokens += 1
    p.money = len(cores) / n
    if cores:
        p.decimal = infer_decimal(cores)
        has_dec = 0
        for c in cores:
            d = parse_core(c, p.decimal)
            if d is not None:
                nums.append(float(d))
            if p.decimal in c and 1 <= len(c) - c.rfind(p.decimal) - 1 <= 2:
                has_dec += 1
        p.money_decimals = has_dec / n
    p.currency_token = cur_tokens / n
    p.integer = sum(bool(re.fullmatch(r"\d+", v)) for v in vals) / n
    p.idshape = sum(id_shaped(v) for v in vals) / n
    lens = [len(v) for v in vals]
    p.len_mean = statistics.fmean(lens)
    p.len_cv = (statistics.pstdev(lens) / p.len_mean) if p.len_mean else 0.0
    p.currency = sum(v.strip().upper() in CURRENCY_CODES or v.strip() in CURRENCY_SYMBOLS for v in vals) / n
    normed = [norm_text(v) for v in vals]
    p.status = sum(t in STATUS_LOOKUP for t in normed) / n
    p.method = sum(t in METHOD_LOOKUP for t in normed) / n
    p.country = sum(t in COUNTRY_LOOKUP for t in normed) / n
    alpha = company = 0
    for v, t in zip(vals, normed):
        letters = sum(ch.isalpha() for ch in v)
        if letters >= 0.5 * len(v.replace(" ", "")) and "@" not in v:
            alpha += 1
        if {w.strip(".,&") for w in t.split()} & COMPANY_SUFFIXES:
            company += 1
    p.alpha, p.company = alpha / n, company / n
    p.words_mean = statistics.fmean(len(v.split()) for v in vals)
    shapes = Counter(shape(v) for v in vals)
    top, cnt = shapes.most_common(1)[0]
    p.shape_top, p.shape_share = top, cnt / n
    if nums:
        p.num_min, p.num_max, p.num_mean = min(nums), max(nums), round(statistics.fmean(nums), 2)
    p.kind = _classify(p)
    p.sketch = sorted(_hash(v) for v in _canonical_values(p, distinct))[:SKETCH_K]
    return p


def _canonical_values(p: Profile, distinct: set[str]) -> set[str]:
    """Sketch dates and amounts by what they mean, so a re-formatted column still overlaps with its old self."""
    if p.kind == "date" and p.date_format:
        return {d.isoformat() for d in (parse_date(v, p.date_format) for v in distinct) if d} or distinct
    if p.kind == "money":
        out = set()
        for v in distinct:
            core, _, neg = split_money(v)
            num = parse_core(core, p.decimal) if core else None
            if num is not None:
                out.add(f"{-num if neg else num:.2f}")
        return out or distinct
    return distinct


def profile_table(table: RawTable) -> list[Profile]:
    return [profile_column(name, table.column(i)) for i, name in enumerate(table.columns)]


# ------------------------------------------------------------ comparisons


def sketch_overlap(a: list[int], b: list[int]) -> float:
    """Jaccard estimate from two bottom-k sketches."""
    if not a or not b:
        return 0.0
    k = min(len(a), len(b), SKETCH_K)
    union = sorted(set(a) | set(b))[:k]
    sa, sb = set(a), set(b)
    return sum(1 for h in union if h in sa and h in sb) / k


def inclusion(inner: frozenset, outer: frozenset) -> float:
    """Share of inner's distinct values found in outer."""
    if not inner:
        return 0.0
    return len(inner & outer) / len(inner)
