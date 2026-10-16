"""Second exports for each tenant: a clean "control" sync and a sync with injected schema drift.

Both contain the first export's rows plus new rows and some changed rows, the way a
nightly full re-export does. The drift sync also has two or three injected events
(renamed, added, dropped or retyped columns). The events are recorded as truth so the
drift detector can be scored on precision and recall, with the control syncs
measuring false alarms.

Event kinds are the generator's words: rename_mild, rename_major, add_unmapped,
add_mappable, drop_optional, drop_required, retype_date, retype_decimal,
retype_cents, id_shape.
"""
from __future__ import annotations

import copy
import json
import random
import re
from decimal import Decimal
from pathlib import Path

from .generate import (CENTS_HEADERS, DATE_FMTS_BY_LEVEL, EXTRA_HEADERS, EXTRAS_BY_ENTITY, HEADERS, ID_START, PAYMENT_ID_FMT,
                       STYLES_BY_LEVEL, ColSpec, FileSpec, TenantSpec, _style_header, _write_sync, _id_fmt)
from .schema import ENTITIES

EVENT_KINDS = ["rename_mild", "rename_major", "add_unmapped", "add_mappable", "drop_optional", "drop_required",
               "retype_date", "retype_decimal", "retype_cents", "id_shape"]


def evolve(base: dict[str, list[dict]], spec: TenantSpec, rng: random.Random) -> dict[str, list[dict]]:
    """New rows and changed rows on top of the first export."""
    new = copy.deepcopy(base)
    cfmt, ifmt, pfmt = (_id_fmt(spec, e) for e in ("customer", "invoice", "payment"))
    n_c = rng.randint(4, 9)
    for j in range(n_c):
        proto = rng.choice(base["customer"])
        c = dict(proto)
        c["customer_id"] = cfmt.format(ID_START["customer"] + 20000 + j)
        c["name"] = f"{proto['name'].split()[0]} Newco {j}"
        c["email"] = f"ap{j}@newco{j}.example"
        new["customer"].append(c)
    n_i = rng.randint(10, 22)
    for j in range(n_i):
        proto = rng.choice(base["invoice"])
        i = dict(proto)
        i["invoice_id"] = ifmt.format(ID_START["invoice"] + 20000 + j)
        i["customer_id"] = rng.choice(new["customer"])["customer_id"]
        i["amount"] = f"{Decimal(rng.randint(5000, 2500000)) / 100:.2f}"
        i["status"] = "open"
        new["invoice"].append(i)
    for inv in new["invoice"]:
        if inv["status"] == "open" and rng.random() < 0.12:
            inv["status"] = rng.choice(["paid", "overdue"])
    for j in range(rng.randint(5, 12)):
        inv = rng.choice(new["invoice"])
        new["payment"].append({"payment_id": pfmt.format(ID_START["payment"] + 20000 + j), "invoice_id": inv["invoice_id"],
                               "payment_date": inv["invoice_date"], "amount": inv["amount"],
                               "method": rng.choice(["card", "ach", "wire", "check", "cash"]), "_cur": inv["currency"]})
    return new


def _restyle(rng: random.Random, header: str, taken: set[str]) -> str:
    toks = [t for t in re.split(r"[_\s\-.]+|(?<=[a-z0-9])(?=[A-Z])", header) if t]
    for _ in range(12):
        h = _style_header(rng, "_".join(toks), rng.choice(["snake", "camel", "pascal", "title", "upper", "upper_space"]), None)
        if h != header and h not in taken:
            return h
    return header + "_x"


def _candidates(spec: TenantSpec) -> list[tuple[str, str, ColSpec | None]]:
    """(event_kind, entity, column or None) that could be applied to this tenant."""
    out = []
    for entity, fs in spec.files.items():
        present = {c.field for c in fs.cols if c.field}
        for c in fs.cols:
            if c.field:
                fdef = next(f for f in ENTITIES[entity] if f.name == c.field)
                out += [("rename_mild", entity, c), ("rename_major", entity, c)]
                out.append(("drop_required" if fdef.required else "drop_optional", entity, c))
                if fdef.ftype == "date":
                    out.append(("retype_date", entity, c))
                if fdef.ftype == "money":
                    out.append(("retype_decimal", entity, c))
                    if c.num_style != "cents" and (entity != "invoice" or c.cur_mode == "column"):
                        out.append(("retype_cents", entity, c))
                if c.field == "payment_id":
                    out.append(("id_shape", entity, c))
            else:
                out.append(("drop_optional", entity, c))
        out.append(("add_unmapped", entity, None))
        if any(f.name not in present and not f.required and not (entity == "invoice" and f.name == "currency") for f in ENTITIES[entity]):
            out.append(("add_mappable", entity, None))
    return out


def apply_events(spec: TenantSpec, base: dict[str, list[dict]], rng: random.Random, n_events: int):
    """Mutate a copy of the spec. Returns (spec2, base2, events)."""
    spec2 = copy.deepcopy(spec)
    base2 = copy.deepcopy(base)
    events: list[dict] = []
    touched: set[int] = set()
    added_to: set[str] = set()
    by_kind: dict[str, list[tuple[str, ColSpec | None]]] = {}
    for k, e, c in _candidates(spec2):
        by_kind.setdefault(k, []).append((e, c))
    used_kinds: list[str] = []
    attempts = 0
    while len(events) < n_events and attempts < 80:
        attempts += 1
        avail = sorted(k for k, v in by_kind.items() if v)
        if not avail:
            break
        fresh = [k for k in avail if k not in used_kinds] or avail      # spread events across kinds
        kind = rng.choice(fresh)
        entity, col = by_kind[kind].pop(rng.randrange(len(by_kind[kind])))
        fs = spec2.files[entity]
        if col is not None:
            if col not in fs.cols or id(col) in touched:
                continue
        elif entity in added_to:
            continue
        taken = {c.source for c in fs.cols}
        ev: dict | None = None
        level = spec2.difficulty
        if kind == "rename_mild":
            old = col.source
            new_leaf = _restyle(rng, col.header, taken)
            col.header = new_leaf
            ev = {"kind": kind, "column": old, "new_column": col.source}
        elif kind == "rename_major":
            old = col.source
            if col.field:
                pool = [h for tier in HEADERS[f"{entity}.{col.field}"].values() for h in tier]
                raw = rng.choice(pool)
                new_leaf = _style_header(rng, raw, rng.choice(STYLES_BY_LEVEL[level]), None)
            else:
                new_leaf = f"X{rng.randint(10, 99)}"
            if new_leaf in taken or new_leaf == col.header:
                new_leaf = f"fld_{rng.randint(100, 999)}"
            col.header = new_leaf
            ev = {"kind": kind, "column": old, "new_column": col.source}
        elif kind == "add_unmapped":
            pool = [x for x in EXTRAS_BY_ENTITY[entity] if x not in {c.extra for c in fs.cols}]
            if not pool:
                continue
            x = rng.choice(pool)
            h = _style_header(rng, rng.choice(EXTRA_HEADERS[x]), "snake", None)
            if h in taken:
                continue
            nc = ColSpec(None, h, extra=x, date_fmt="%Y-%m-%d", num_style="plain", cur_mode="column")
            fs.cols.insert(rng.randint(0, len(fs.cols)), nc)
            added_to.add(entity)
            ev = {"kind": kind, "column": nc.source}
        elif kind == "add_mappable":
            present = {c.field for c in fs.cols if c.field}
            options = [f for f in ENTITIES[entity] if f.name not in present and not f.required and not (entity == "invoice" and f.name == "currency")]
            if not options:
                continue
            f = rng.choice(options)
            raw = rng.choice(HEADERS[f"{entity}.{f.name}"]["core"])
            h = _style_header(rng, raw, "snake", None)
            if h in taken:
                continue
            nc = ColSpec(f.name, h, date_fmt="%Y-%m-%d", num_style="plain", status_lang=spec2.lang, method_lang=spec2.lang)
            fs.cols.insert(rng.randint(0, len(fs.cols)), nc)
            added_to.add(entity)
            ev = {"kind": kind, "column": nc.source, "field": f.name}
        elif kind in ("drop_optional", "drop_required"):
            ev = {"kind": kind, "column": col.source}
            fs.cols.remove(col)
        elif kind == "retype_date":
            options = [f for f in DATE_FMTS_BY_LEVEL["medium"] if f != col.date_fmt]
            col.date_fmt = rng.choice(options)
            ev = {"kind": kind, "column": col.source, "to": col.date_fmt}
        elif kind == "retype_decimal":
            if col.num_style == "cents":
                continue
            col.num_style = "eu" if col.num_style in ("plain", "us", "swiss") else "us"      # always flips the decimal separator
            col.json_numeric = False
            ev = {"kind": kind, "column": col.source, "to": col.num_style}
        elif kind == "retype_cents":
            col.num_style = "cents"
            col.cur_mode = "column"
            col.json_numeric = False
            ev = {"kind": kind, "column": col.source}
        elif kind == "id_shape":
            old_fmt = col.id_fmt
            col.id_fmt = rng.choice([f for f in PAYMENT_ID_FMT if f != old_fmt])
            for p in base2["payment"]:
                n = int(re.search(r"(\d+)$", p["payment_id"]).group(1))
                p["payment_id"] = col.id_fmt.format(n)
            ev = {"kind": kind, "column": col.source, "to": col.id_fmt}
        if ev is None:
            continue
        if col is not None:
            touched.add(id(col))
        used_kinds.append(kind)
        ev.update({"file": fs.filename, "entity": entity})
        if col is not None and col.field:
            ev["field"] = col.field
        events.append(ev)
    return spec2, base2, events


def make_syncs(root: Path, spec: TenantSpec, base: dict[str, list[dict]], seed: int) -> None:
    rng = random.Random(f"evolve:{seed}:{spec.tid}")
    evolved = evolve(base, spec, rng)
    # control: same shape, new data
    truth = _write_sync(root, "control", spec, evolved, "control")
    truth["events"] = []
    (root / "truth_control.json").write_text(json.dumps(truth, indent=1, ensure_ascii=False), encoding="utf-8")
    # drift: events applied
    spec2, evolved2, events = apply_events(spec, evolved, rng, rng.choice([3, 4]))
    truth = _write_sync(root, "drift", spec2, evolved2, "drift")
    truth["events"] = events
    (root / "truth_drift.json").write_text(json.dumps(truth, indent=1, ensure_ascii=False), encoding="utf-8")
