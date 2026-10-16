"""Tables and charts, generated from results/summary.json so every number in the README is read from a file."""
from __future__ import annotations

import json
from pathlib import Path

PALETTE = {"naive (exact header)": "#999999", "header only": "#E69F00", "value only": "#56B4E9", "header + value": "#009E73",
           "tuned (all signals)": "#0072B2"}
MARK_START, MARK_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def charts(summary: dict, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    out = []
    m = summary["mapping"]
    systems = list(m)
    diffs = ["easy", "medium", "hard"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    width = 0.16
    for i, s in enumerate(systems):
        vals = [m[s]["by_difficulty"].get(d, {}).get("f1", 0) for d in diffs]
        xs = [j + (i - (len(systems) - 1) / 2) * width for j in range(len(diffs))]
        bars = ax.bar(xs, vals, width, label=s, color=PALETTE.get(s, "#444"))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(diffs)), [f"{d}\n({m[systems[0]]['by_difficulty'][d]['tenants']} tenants)" for d in diffs])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("column mapping F1 (held-out tenants)")
    ax.set_title("Combined evidence holds up on the hard tenants; any single signal does not")
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    p = out_dir / "mapping_by_difficulty.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)

    abl = summary["ablation"]
    names = sorted(abl, key=lambda k: abl[k]["delta_vs_full"])
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(names) + 1.4))
    ys = range(len(names))
    deltas = [abl[k]["delta_vs_full"] for k in names]
    lo = [abl[k]["delta_vs_full"] - abl[k]["delta_ci"][0] for k in names]
    hi = [abl[k]["delta_ci"][1] - abl[k]["delta_vs_full"] for k in names]
    ax.barh(list(ys), deltas, xerr=[lo, hi], color="#0072B2", error_kw={"ecolor": "#444", "lw": 1})
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_yticks(list(ys), names)
    ax.set_xlabel("change in held-out F1 when this part is removed (bars: 95% interval over tenants)")
    ax.set_title("What each part of the mapper is worth")
    fig.tight_layout()
    p = out_dir / "ablation.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)

    d = summary["drift"]["predicted_mapping"]
    kinds = list(d["recall_by_kind"])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    vals = [d["recall_by_kind"][k]["found"] / d["recall_by_kind"][k]["of"] for k in kinds]
    ax.barh(kinds, vals, color="#009E73")
    for i, k in enumerate(kinds):
        ax.text(vals[i] + 0.01, i, f"{d['recall_by_kind'][k]['found']}/{d['recall_by_kind'][k]['of']}", va="center", fontsize=8)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("share of injected events detected (held-out tenants)")
    ax.set_title(f"Drift detection: precision {_pct(d['precision'])}, {d['control_false_alarms']} false alarms on {d['control_syncs']} clean re-exports")
    fig.tight_layout()
    p = out_dir / "drift.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)
    return out


def results_markdown(summary: dict) -> str:
    meta, m = summary["meta"], summary["mapping"]
    cov = meta["lexicon_coverage"]
    L: list[str] = []
    L += ["![Mapping F1 by difficulty](results/mapping_by_difficulty.png)", "", "![What each part is worth](results/ablation.png)", "",
          "![Drift detection](results/drift.png)", ""]
    L += ["### Setup", "", "| | |", "|---|---|",
          f"| Tenants | {meta['tenants']} synthetic: {len(meta['tuning'])} tuning, {len(meta['heldout'])} held out (split once, seed {meta['split_seed']}, stratified by difficulty) |",
          f"| Generator seed | {meta['gen_seed']} |",
          f"| Mapped columns whose header has no word the lexicon knows | {cov['no_known_token']} of {cov['columns']} ({_pct(cov['share'])}), all tenants |",
          f"| Tuned on | tuning tenants only; settings in `results/tuned_params.json`, every trial in `results/trials.jsonl` |",
          f"| Python | {meta['python']} |", "",
          "### Column mapping on held-out tenants", "",
          "Micro-averaged over columns. Interval: 95%, resampling whole tenants. Each simpler system had its own threshold tuned on the tuning tenants.", "",
          "| System | Precision | Recall | F1 (95% interval) | Entity right | easy F1 | medium F1 | hard F1 |", "|---|---|---|---|---|---|---|---|"]
    for name, r in m.items():
        o, bd = r["overall"], r["by_difficulty"]
        lo, hi = r["f1_ci"]
        L.append(f"| {name} | {o['precision']:.3f} | {o['recall']:.3f} | {o['f1']:.3f} ({lo:.3f} to {hi:.3f}) | {_pct(o['entity_accuracy'])} | "
                 + " | ".join(f"{bd[d]['f1']:.3f}" for d in ("easy", "medium", "hard")) + " |")
    L += ["", "The same systems on the tuning tenants (the data the settings were chosen on, so optimistic): "
          + ", ".join(f"{k} {v['overall']['f1']:.3f}" for k, v in summary["mapping_tuning_split"].items()) + ".", ""]
    L += ["### What each part is worth", "", "Held-out F1 with one part of the tuned mapper removed.", "",
          "| Removed | F1 | Change | 95% interval of change |", "|---|---|---|---|"]
    for k, a in sorted(summary["ablation"].items(), key=lambda kv: kv[1]["delta_vs_full"]):
        L.append(f"| {k} | {a['f1']:.3f} | {a['delta_vs_full']:+.3f} | {a['delta_ci'][0]:+.3f} to {a['delta_ci'][1]:+.3f} |")
    r = summary["review"]
    L += ["", "### Human review", "",
          f"The mapper flags {r['flagged_for_review']} of {r['columns_mapped']} mapped columns ({_pct(r['flagged_share'])}) for a person to look at "
          f"(low confidence, a close second choice, or decided by a tie-break rule). {_pct(r['wrong_caught_by_flags'])} of the {r['wrong_predictions']} wrong predictions are among the flagged columns, "
          f"and {_pct(1 - r['precision_of_unflagged'])} of the unflagged columns are wrong.", ""]
    n = summary["normalization"]
    L += ["### Normalization", "",
          "Records found and field values equal to the truth, on the held-out tenants' first export. 'Oracle mapping' uses the true mapping, so it isolates the transform layer; "
          "'predicted mapping' adds the mapper's mistakes.", "",
          "| | Records found | Field accuracy | Rows loaded that should not be | Junk rows held back |", "|---|---|---|---|---|"]
    for k, label in (("oracle_mapping", "oracle mapping"), ("predicted_mapping", "predicted mapping")):
        x = n[k]
        L.append(f"| {label} | {_pct(x['record_recall'])} | {_pct(x['field_accuracy'])} | {x['false_loads']} | {_pct(x['junk_rows_held_back'])} |")
    errs = ", ".join(f"{k} {v}" for k, v in list(n["oracle_mapping"]["field_errors"].items())[:6])
    L += ["", f"Wrong field values with the oracle mapping, by field: {errs}. These are the limits of the transform layer: status and payment-method words and country names outside "
          "the vocabulary, and amounts in minor units (cents) in files with no header that says so.", ""]
    d = summary["drift"]
    L += ["### Schema drift", "",
          f"Each held-out tenant has a second export with 3 or 4 injected changes (renamed, added, dropped, retyped columns) and a clean re-export with new and changed rows but no schema change. "
          f"Rename threshold {d['rename_threshold']}, chosen on the tuning tenants.", "",
          "| Config built from | Events found | Precision | Recall | Syncs decided right | Unsafe misses | False alarms on clean re-exports |", "|---|---|---|---|---|---|---|"]
    for k, label in (("oracle_mapping", "oracle mapping"), ("predicted_mapping", "predicted mapping")):
        x = d[k]
        L.append(f"| {label} | {x['events_true']} | {x['precision']:.3f} | {x['recall']:.3f} | {_pct(x['sync_decision_accuracy'])} | "
                 f"{x['unsafe_misses']} of {x['should_block']} | {x['control_false_alarms']} events, {x['control_false_blocks']} false blocks ({x['control_syncs']} syncs) |")
    x = d["predicted_mapping"]
    miss = ", ".join(f"{k} {v['found']}/{v['of']}" for k, v in x["recall_by_kind"].items())
    c = x["ungated_load_would_corrupt"]
    L += ["", f"Detected by kind (predicted mapping): {miss}.", "",
          f"In the {c['syncs']} exports that should be blocked and were, loading them with the old config would have written {c['fields_wrong']:,} wrong or missing field values "
          f"out of {c['fields']:,} ({_pct(c['fields_wrong'] / max(c['fields'], 1))}).", ""]
    g = summary["ingest"]
    L += ["### Ingest guarantees", "",
          f"Run on the held-out tenants with the true mapping, so the ledger is tested rather than the mapper: {g['passed']} of {g['checks']} checks pass.", "",
          "| Check | Passed | Scenario actually occurred |", "|---|---|---|"]
    for k, v in g["by_check"].items():
        L.append(f"| {k.replace('_', ' ')} | {v['passed']} of {v['of']} | {v['scenario_occurred']} of {v['of']} |")
    if g["failures"]:
        L += ["", "Failures: " + "; ".join(f"{f['tenant']} {f['check']} ({f['detail']})" for f in g["failures"])]
    return "\n".join(L) + "\n"


def write_report(summary_path: Path, out_dir: Path, readme: Path | None = None) -> None:
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    charts(summary, out_dir)
    md = results_markdown(summary)
    (out_dir / "REPORT.md").write_text(md, encoding="utf-8")
    if readme:
        text = readme.read_text(encoding="utf-8")
        a, b = text.index(MARK_START), text.index(MARK_END)
        readme.write_text(text[: a + len(MARK_START)] + "\n\n" + md + "\n" + text[b:], encoding="utf-8")
