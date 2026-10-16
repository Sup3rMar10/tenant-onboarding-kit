"""Read a customer's export without being told how it was written.

Handles CSV/TSV and other delimited text (encoding and delimiter are detected, title
lines above the header and footers below the data are skipped, repeated header rows
from paginated exports are dropped) and JSON, either a list of records, a wrapper
object around one, or newline-delimited. Nested JSON is flattened to dotted paths.

Excel files are not supported; ask the customer for CSV.
"""
from __future__ import annotations

import codecs
import csv
import io
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

csv.field_size_limit(10_000_000)

_WRAPPER_KEYS = ("data", "records", "items", "rows", "results", "invoices", "customers", "payments")


@dataclass
class RawTable:
    name: str
    columns: list[str]
    rows: list[list[str]]
    meta: dict = field(default_factory=dict)

    def column(self, i: int) -> list[str]:
        return [r[i] for r in self.rows]


def decode(data: bytes) -> tuple[str, str]:
    if data.startswith(codecs.BOM_UTF8):
        return data[3:].decode("utf-8"), "utf-8-sig"
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16"), "utf-16"
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass
    try:
        return data.decode("cp1252"), "cp1252"
    except UnicodeDecodeError:
        return data.decode("latin-1"), "latin-1"


def sniff_delimiter(text: str) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()][:60]
    best, best_score = ",", (-1.0, -1)
    for d in (",", ";", "\t", "|"):
        counts = []
        for ln in lines:
            try:
                counts.append(len(next(csv.reader([ln], delimiter=d))))
            except (csv.Error, StopIteration):
                counts.append(0)
        if not counts:
            continue
        modal, n = Counter(counts).most_common(1)[0]
        score = (n / len(counts) if modal > 1 else 0.0, modal)
        if score > best_score:
            best, best_score = d, score
    return best


def _header_like(row: list[str]) -> bool:
    cells = [c.strip() for c in row if c.strip()]
    if len(cells) < max(2, len(row) // 2) or len(set(cells)) < 2:
        return False
    texty = sum(1 for c in cells if re.search(r"[^\W\d_]", c) and not re.fullmatch(r"[\d.,/\-: ]+", c))
    return texty >= 0.6 * len(cells)


def _unique(names: list[str]) -> list[str]:
    seen: Counter = Counter()
    out = []
    for n in names:
        seen[n] += 1
        out.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    return out


def read_delimited(data: bytes, name: str = "") -> RawTable:
    text, encoding = decode(data)
    text = text.lstrip("﻿")
    delim = sniff_delimiter(text)
    rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=delim))
    lens = Counter(len(r) for r in rows if len(r) > 1) or Counter(len(r) for r in rows)
    modal = lens.most_common(1)[0][0] if lens else 0
    header_idx = next((i for i, r in enumerate(rows) if len(r) == modal and _header_like(r)), None)
    if header_idx is None:
        raise ValueError(f"{name}: could not find a header row")
    header = [h.strip().lstrip("﻿") for h in rows[header_idx]]
    dropped: Counter = Counter()
    dropped["title_lines"] = header_idx
    out_rows: list[list[str]] = []
    for r in rows[header_idx + 1:]:
        cells = [c.strip() for c in r]
        if not any(cells):
            dropped["blank"] += 1
        elif cells == header:
            dropped["repeated_header"] += 1
        elif len(cells) != modal:
            if len(cells) == 1 or sum(1 for c in cells if c) <= 1:
                dropped["footer"] += 1
            elif len(cells) < modal and sum(1 for c in cells if c) >= modal / 2:
                out_rows.append(cells + [""] * (modal - len(cells)))
            else:
                dropped["malformed"] += 1
        else:
            out_rows.append(cells)
    return RawTable(name, _unique(header), out_rows,
                    {"kind": "delimited", "encoding": encoding, "delimiter": delim, "header_row": header_idx, "dropped": dict(dropped)})


def _flatten(obj, prefix: str, out: dict) -> None:
    for k, v in obj.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            _flatten(v, key, out)
        elif isinstance(v, list):
            out[key] = json.dumps(v, ensure_ascii=False)
        else:
            out[key] = v


def _scalar(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v).strip()


def _records(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for k in _WRAPPER_KEYS:
            if isinstance(payload.get(k), list):
                return payload[k]
        for v in payload.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
        return [payload]
    raise ValueError("unrecognised JSON layout")


def read_json(data: bytes, name: str = "", ndjson: bool = False) -> RawTable:
    text, encoding = decode(data)
    text = text.lstrip("﻿")
    records, kind = None, "json"
    if not ndjson:
        try:
            records = _records(json.loads(text))
        except json.JSONDecodeError:
            records = None
    if records is None:
        records, kind = [json.loads(ln) for ln in text.splitlines() if ln.strip()], "ndjson"
    flat = []
    for rec in records:
        row: dict = {}
        if isinstance(rec, dict):
            _flatten(rec, "", row)
        flat.append(row)
    columns: list[str] = []
    for row in flat:
        for k in row:
            if k not in columns:
                columns.append(k)
    rows = [[_scalar(row.get(c)) for c in columns] for row in flat]
    nested = any("." in c for c in columns)
    return RawTable(name, columns, rows, {"kind": kind, "encoding": encoding, "nested": nested, "dropped": {}})


def read_table(path: Path | str) -> RawTable:
    path = Path(path)
    data = path.read_bytes()
    suffix = path.suffix.lower()
    if suffix in (".json", ".ndjson", ".jsonl"):
        return read_json(data, path.name, ndjson=suffix in (".ndjson", ".jsonl"))
    if suffix in (".xlsx", ".xls"):
        raise ValueError(f"{path.name}: Excel files are not supported, export to CSV")
    return read_delimited(data, path.name)
