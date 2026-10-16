"""Map a tenant's columns to the canonical schema.

Three kinds of evidence, each scored 0..1 for every (column, canonical field):

  header    how close the column name is to what the lexicon and fuzzy matching know.
  value     does the content look like that field's type (email, date, amount, status word...).
  relation  facts that only show up across columns and files: a key is unique and gets
            referenced by another file; a foreign key's values all appear in another
            file's key column.

Scores are combined as a weighted average over the evidence that exists for that
column (a column whose name matches nothing in the lexicon drops the header term, rather
than being punished for it). Each file is assigned to the entity whose fields it fits
best, columns are assigned one-to-one to fields above a threshold, and two small rules
settle ties that need row-level facts (which of several date columns is the issue date,
which of several amounts is the total). Anything unassigned stays unmapped.

The weights and threshold are tuned in tune.py on the tuning tenants only.
An optional arbiter (llm.py) is asked about the few columns the rules leave ambiguous.
"""
from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import date
from functools import lru_cache
from difflib import SequenceMatcher
from statistics import median

from .parsing import parse_date
from .profile import Profile, inclusion, profile_table
from .readers import RawTable
from .schema import ENTITIES, ENTITY_WORDS, IGNORE_TOKENS, LEXICON, Field, all_targets, header_tokens, norm_text

HEADER_INFORMATIVE = 0.4          # below this, a column's name carries no evidence about any field
KEY_UNIQUE = 0.9


@dataclass(frozen=True)
class Params:
    """The defaults are what tune.py chose on the tuning tenants (results/tuned_params.json; a test keeps them in step)."""

    w_header: float = 1.0
    w_value: float = 1.0
    w_rel: float = 0.5
    w_file: float = 0.0
    threshold: float = 0.35
    header_prior: float = 0.55         # what an uninformative column name counts for, instead of 0
    one_per_entity: bool = True        # a tenant exports each entity once (soft assumption, see README)
    ignore_factor: float = 1.0         # multiplier when the name says "audit stamp / hash / note"; 1.0 is off
    date_rule: bool = True
    magnitude_rule: bool = True
    residual_rule: bool = False
    review_below: float = 0.7
    review_margin: float = 0.1

    def to_dict(self) -> dict:
        return dict(self.__dict__)


DEFAULT_PARAMS = Params()


# ------------------------------------------------------------------ header score


@lru_cache(maxsize=None)
def _ratio(a: str, b: str) -> float:
    sm = SequenceMatcher(None, a, b)
    if sm.real_quick_ratio() < 0.7 or sm.quick_ratio() < 0.7:
        return 0.0
    return sm.ratio()


@lru_cache(maxsize=None)
def header_score(tokens: tuple[str, ...], key: str) -> float:
    if not tokens:
        return 0.0
    joined = "".join(tokens)
    tset = set(tokens)
    best = 0.0
    for ptoks, w in LEXICON[key]:
        pj = "".join(ptoks)
        if tokens == ptoks:
            s = 1.0
        elif joined == pj:
            s = 0.97
        elif set(ptoks) <= tset:
            s = 0.9 if len(ptoks) > 1 else 0.8
        elif tset <= set(ptoks):
            s = 0.6
        else:
            r = _ratio(joined, pj)
            s = 0.8 * r if r >= 0.75 else 0.0
        best = max(best, s * w)
    return best


def column_header_scores(source: str) -> dict[str, float]:
    leaf = header_tokens(source.split(".")[-1])
    full = header_tokens(source)
    out = {}
    for e, f in all_targets():
        key = f"{e}.{f}"
        out[key] = max(header_score(leaf, key), 0.95 * header_score(full, key))
    return out


# ------------------------------------------------------------------ value score


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def value_fit(p: Profile, f: Field) -> float:
    if p.n_nonempty == 0:
        return 0.0
    t = f.ftype
    if t == "email":
        return p.email
    if t == "phone":
        return p.phone * (1 - p.date) * (1 - p.money_decimals)       # 2025-03-04 and 13524.82 are phone-shaped too
    if t == "country":
        return p.country
    if t == "date":
        return p.date
    if t == "currency":
        return p.currency
    if t == "status":
        return p.status
    if t == "method":
        return p.method
    if t == "money":
        id_like = p.integer > 0.95 and p.len_cv <= 0.1 and p.uniq_ratio > 0.9
        base = p.money * (0.55 + 0.45 * min(1.0, p.money_decimals / 0.5)) * (1 - 0.5 * id_like)
        return base * (1 - p.date)
    if t == "text":
        uniq_gate = _clamp(p.uniq_ratio / 0.7)
        return p.alpha * (1 - p.email) * (1 - p.date) * uniq_gate * (0.55 + 0.45 * min(1.0, p.company / 0.5))
    if t == "id":
        return p.idshape * (1 - p.date) * (1 - p.money_decimals)
    return 0.0


# --------------------------------------------------------------------- signals


@dataclass
class ColSignals:
    source: str
    profile: Profile
    header: dict[str, float]
    value: dict[str, float]
    header_informative: bool
    ignore: bool = False
    refs_out: list[tuple[float, int, int]] = field(default_factory=list)   # (inclusion, other file, other column)
    refd_in: float = 0.0


@dataclass
class FileSignals:
    name: str
    columns: list[ColSignals]
    name_hint: dict[str, float]
    dates: dict[int, list[date | None]] = field(default_factory=dict)       # parsed rows of date-like columns
    n_rows: int = 0


@dataclass
class TenantSignals:
    files: list[FileSignals]
    tables: list[RawTable]


def _name_hint(filename: str) -> dict[str, float]:
    stem = filename.rsplit(".", 1)[0]
    toks = set(header_tokens(stem))
    return {e: 1.0 if toks & words else 0.0 for e, words in ENTITY_WORDS.items()}


def compute_signals(tables: list[RawTable]) -> TenantSignals:
    files: list[FileSignals] = []
    for t in tables:
        profs = profile_table(t)
        cols = []
        for src, p in zip(t.columns, profs):
            hs = column_header_scores(src)
            vs = {f"{e}.{f.name}": value_fit(p, f) for e, fs in ENTITIES.items() for f in fs}
            cols.append(ColSignals(src, p, hs, vs, max(hs.values(), default=0.0) >= HEADER_INFORMATIVE,
                                   ignore=bool(set(header_tokens(src.split('.')[-1])) & IGNORE_TOKENS)))
        fsig = FileSignals(t.name, cols, _name_hint(t.name), n_rows=len(t.rows))
        for i, c in enumerate(cols):
            if c.profile.kind == "date" and c.profile.date_format:
                fsig.dates[i] = [parse_date(v, c.profile.date_format) if v else None for v in t.column(i)]
        files.append(fsig)
    # relations across files
    uniq = [(g, j) for g, fs in enumerate(files) for j, c in enumerate(fs.columns)
            if c.profile.uniq_ratio >= KEY_UNIQUE and c.profile.n_distinct >= 5]
    for f, fs in enumerate(files):
        for i, c in enumerate(fs.columns):
            if c.profile.n_distinct < 5 or c.profile.kind in ("date", "email", "phone", "money", "status", "method", "country", "currency"):
                continue
            for g, j in uniq:
                if g == f:
                    continue
                inc = inclusion(c.profile.values, files[g].columns[j].profile.values)
                if inc >= 0.8:
                    c.refs_out.append((inc, g, j))
                    files[g].columns[j].refd_in = max(files[g].columns[j].refd_in, inc)
    return TenantSignals(files, tables)


# -------------------------------------------------------------------- combining


def relation_score(c: ColSignals, f: Field, entity: str, file_entity: dict[int, str] | None) -> float | None:
    if f.ftype != "id":
        return None
    p = c.profile
    refs = c.refs_out
    if f.ref:                                           # foreign key: values live in the referenced entity's key column
        if file_entity is not None:
            refs = [r for r in refs if file_entity.get(r[1]) == f.ref]
        return max((r[0] for r in refs), default=0.0)
    uniq_score = _clamp((p.uniq_ratio - 0.5) / 0.47)    # key: unique, not pointing elsewhere, ideally pointed at
    out_inc = max((r[0] for r in c.refs_out), default=0.0)
    return uniq_score * (1 - 0.8 * out_inc) * (0.75 + 0.25 * c.refd_in)


def field_score(c: ColSignals, entity: str, f: Field, params: Params, file_entity: dict[int, str] | None) -> float:
    key = f"{entity}.{f.name}"
    parts = []
    if params.w_header > 0:
        parts.append((params.w_header, c.header[key] if c.header_informative else params.header_prior))
    if params.w_value > 0:
        parts.append((params.w_value, c.value[key]))
    r = relation_score(c, f, entity, file_entity)
    if r is not None and params.w_rel > 0:
        parts.append((params.w_rel, r))
    if not parts:
        return 0.0
    wsum = sum(w for w, _ in parts)
    score = sum(w * s for w, s in parts) / wsum
    return score * params.ignore_factor if c.ignore else score


@dataclass
class ColumnMapping:
    source: str
    field: str | None
    score: float = 0.0
    margin: float = 0.0
    review: bool = False
    evidence: dict = field(default_factory=dict)
    alternatives: list[tuple[str, float]] = field(default_factory=list)
    decided_by: str = "score"          # score | date_rule | magnitude_rule | arbiter


@dataclass
class FileMapping:
    filename: str
    entity: str
    entity_scores: dict[str, float]
    columns: list[ColumnMapping]

    def by_source(self) -> dict[str, ColumnMapping]:
        return {c.source: c for c in self.columns}


@dataclass
class TenantMapping:
    files: dict[str, FileMapping]

    def predictions(self) -> set[tuple[str, str, str, str]]:
        return {(fn, c.source, fm.entity, c.field) for fn, fm in self.files.items() for c in fm.columns if c.field}


def _assign(fs: FileSignals, entity: str, params: Params, file_entity) -> tuple[dict[int, tuple[str, float]], float]:
    cand = []
    for i, c in enumerate(fs.columns):
        for f in ENTITIES[entity]:
            sc = field_score(c, entity, f, params, file_entity)
            cand.append((sc, i, f.name, sc + (0.02 if f.required else 0.0)))
    cand.sort(key=lambda x: (-x[3], x[1], x[2]))
    taken_c: set[int] = set()
    taken_f: set[str] = set()
    out: dict[int, tuple[str, float]] = {}
    for s, i, fname, _ in cand:
        if s < params.threshold:
            continue
        if i in taken_c or fname in taken_f:
            continue
        out[i] = (fname, s)
        taken_c.add(i)
        taken_f.add(fname)
    excess = sum(s - params.threshold for _, s in out.values())
    return out, excess


def _median_date(ds: list[date | None]) -> float:
    xs = [d.toordinal() for d in ds if d]
    return median(xs) if xs else float("inf")


def _terms_like(a: list[date | None], b: list[date | None]) -> float:
    offs = [(y - x).days for x, y in zip(a, b) if x and y]
    if len(offs) < 5:
        return 0.0
    top = sum(n for _, n in Counter(offs).most_common(4)) / len(offs)
    return top if min(offs) >= 0 else 0.0


def _apply_rules(fs: FileSignals, entity: str, assign: dict[int, tuple[str, float]], params: Params) -> dict[int, str]:
    """Return {column index: reason} for the columns the rules re-assigned."""
    changed: dict[int, str] = {}
    primary = {"customer": "created_date", "invoice": "invoice_date", "payment": "payment_date"}[entity]
    if params.date_rule:
        date_fields = {f.name for f in ENTITIES[entity] if f.ftype == "date"}
        # date columns whose names say nothing about which date they are
        U = [i for i, c in enumerate(fs.columns)
             if i in fs.dates and (not c.header_informative or max(c.header[f"{entity}.{d}"] for d in date_fields) < 0.5)]
        U = [i for i in U if i not in assign or assign[i][0] in date_fields]
        if len(U) >= 2:
            U.sort(key=lambda i: _median_date(fs.dates[i]))
            first = U[0]
            for i in U:
                assign.pop(i, None)
            for k in [k for k, (fname, _) in assign.items() if fname in date_fields and fname in (primary, "due_date")]:
                if k in U:
                    assign.pop(k, None)
            taken = {fname for fname, _ in assign.values()}
            if primary not in taken:
                assign[first] = (primary, 0.6)
                changed[first] = "date_rule"
            if entity == "invoice" and "due_date" not in taken:
                rest = [i for i in U[1:] if i not in assign]
                if rest:
                    best = max(rest, key=lambda i: _terms_like(fs.dates[first], fs.dates[i]))
                    if _terms_like(fs.dates[first], fs.dates[best]) >= 0.6:
                        assign[best] = ("due_date", 0.6)
                        changed[best] = "date_rule"
    if params.magnitude_rule and entity in ("invoice", "payment"):
        M = [i for i, c in enumerate(fs.columns)
             if c.profile.kind in ("money", "int") and c.profile.num_mean is not None
             and (not c.header_informative or c.header[f"{entity}.amount"] < 0.5) and (i not in assign or assign[i][0] == "amount")]
        if len(M) >= 2:
            biggest = max(M, key=lambda i: fs.columns[i].profile.num_mean)
            for i in M:
                if i in assign and assign[i][0] == "amount":
                    assign.pop(i)
            if "amount" not in {f for f, _ in assign.values()}:
                assign[biggest] = ("amount", 0.6)
                changed[biggest] = "magnitude_rule"
    if params.residual_rule:
        taken = {f for f, _ in assign.values()}
        for f in ENTITIES[entity]:
            if f.ftype in ("status", "method") and f.name not in taken:
                cands = [i for i, c in enumerate(fs.columns)
                         if i not in assign and not c.ignore and not c.header_informative and c.profile.kind in ("text", "status", "method")
                         and 2 <= c.profile.n_distinct <= 8 and c.profile.uniq_ratio <= 0.2 and c.profile.len_mean <= 16]
                if len(cands) == 1:
                    assign[cands[0]] = (f.name, 0.45)
                    changed[cands[0]] = "residual_rule"
                    taken.add(f.name)
    return changed


def map_file_as(fs: FileSignals, entity: str, params: Params, file_entity: dict[int, str] | None, entity_scores: dict[str, float]) -> FileMapping:
    assign, _ = _assign(fs, entity, params, file_entity)
    assign = dict(assign)
    changed = _apply_rules(fs, entity, assign, params)
    cols = []
    for i, c in enumerate(fs.columns):
        alts = sorted(((f"{entity}.{f.name}", field_score(c, entity, f, params, file_entity)) for f in ENTITIES[entity]), key=lambda x: -x[1])
        if i in assign:
            fname, s = assign[i]
            others = [a for k, a in alts if k != f"{entity}.{fname}"]
            margin = s - (others[0] if others else 0.0)
            review = s < params.review_below or margin < params.review_margin or i in changed
            cols.append(ColumnMapping(c.source, fname, round(s, 4), round(margin, 4), review,
                                      {"header": round(c.header[f"{entity}.{fname}"], 3), "value": round(c.value[f"{entity}.{fname}"], 3)},
                                      [(k, round(a, 3)) for k, a in alts[:3]], changed.get(i, "score")))
        else:
            cols.append(ColumnMapping(c.source, None, 0.0, 0.0, False, {}, [(k, round(a, 3)) for k, a in alts[:2]]))
    return FileMapping(fs.name, entity, entity_scores, cols)


def choose_entities(sig: TenantSignals, params: Params) -> tuple[dict[int, str], list[dict[str, float]]]:
    n = len(sig.files)
    solo: list[dict[str, float]] = []
    for fs in sig.files:
        solo.append({e: round(_assign(fs, e, params, None)[1] + params.w_file * fs.name_hint[e], 4) for e in ENTITIES})
    if params.one_per_entity and n <= len(ENTITIES):
        best = None
        for perm in itertools.permutations(ENTITIES, n):
            ctx = dict(enumerate(perm))
            tot = sum(_assign(fs, perm[k], params, ctx)[1] + params.w_file * fs.name_hint[perm[k]] for k, fs in enumerate(sig.files))
            if best is None or tot > best[0] + 1e-9:
                best = (tot, perm)
        return dict(enumerate(best[1])), solo
    return {k: max(d, key=lambda e: (d[e], e)) for k, d in enumerate(solo)}, solo


def map_tenant(sig: TenantSignals, params: Params = DEFAULT_PARAMS, arbiter=None) -> TenantMapping:
    chosen, scores = choose_entities(sig, params)
    out = {fs.name: map_file_as(fs, chosen[k], params, chosen, scores[k]) for k, fs in enumerate(sig.files)}
    mapping = TenantMapping(out)
    if arbiter is not None:
        from .llm import consult_arbiter
        consult_arbiter(sig, mapping, params, arbiter)
    return mapping


def map_tables(tables: list[RawTable], params: Params = DEFAULT_PARAMS, arbiter=None) -> TenantMapping:
    return map_tenant(compute_signals(tables), params, arbiter)


# --------------------------------------------------------------------- baseline


_CANON_NAMES = {f.name for fs in ENTITIES.values() for f in fs}


def naive_map(tables: list[RawTable]) -> TenantMapping:
    """The baseline: a column is mapped only if its name, lower-cased with separators removed, is exactly
    a canonical field name, and a file's entity comes only from its file name."""
    out = {}
    for t in tables:
        toks = set(header_tokens(t.name.rsplit(".", 1)[0]))
        hint = {e: float(bool(toks & {e, e + "s"})) for e in ENTITIES}
        entity = max(hint, key=lambda e: (hint[e], e == "customer"))
        names = {f.name.replace("_", ""): f.name for f in ENTITIES[entity]}
        cols = []
        for src in t.columns:
            key = "".join(header_tokens(src.split(".")[-1]))
            cols.append(ColumnMapping(src, names.get(key)))
        out[t.name] = FileMapping(t.name, entity, {}, cols)
    return TenantMapping(out)
