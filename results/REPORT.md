![Mapping F1 by difficulty](results/mapping_by_difficulty.png)

![What each part is worth](results/ablation.png)

![Drift detection](results/drift.png)

### Setup

| | |
|---|---|
| Tenants | 40 synthetic: 20 tuning, 20 held out (split once, seed 20261016, stratified by difficulty) |
| Generator seed | 7 |
| Mapped columns whose header has no word the lexicon knows | 157 of 659 (23.8%), all tenants |
| Tuned on | tuning tenants only; settings in `results/tuned_params.json`, every trial in `results/trials.jsonl` |
| Python | 3.13.13 |

### Column mapping on held-out tenants

Micro-averaged over columns. Interval: 95%, resampling whole tenants. Each simpler system had its own threshold tuned on the tuning tenants.

| System | Precision | Recall | F1 (95% interval) | Entity right | easy F1 | medium F1 | hard F1 |
|---|---|---|---|---|---|---|---|
| naive (exact header) | 1.000 | 0.119 | 0.212 (0.124 to 0.297) | 66.7% | 0.371 | 0.159 | 0.022 |
| header only | 0.952 | 0.787 | 0.861 (0.776 to 0.932) | 91.7% | 1.000 | 0.890 | 0.563 |
| value only | 0.708 | 0.723 | 0.715 (0.654 to 0.777) | 93.3% | 0.813 | 0.701 | 0.601 |
| header + value | 0.892 | 0.908 | 0.900 (0.831 to 0.961) | 95.0% | 1.000 | 0.901 | 0.765 |
| tuned (all signals) | 0.988 | 0.991 | 0.989 (0.978 to 0.999) | 100.0% | 1.000 | 0.991 | 0.972 |

The same systems on the tuning tenants (the data the settings were chosen on, so optimistic): naive (exact header) 0.145, header only 0.824, value only 0.705, header + value 0.917, tuned (all signals) 0.989.

### What each part is worth

Held-out F1 with one part of the tuned mapper removed.

| Removed | F1 | Change | 95% interval of change |
|---|---|---|---|
| no value signal | 0.844 | -0.146 | -0.224 to -0.079 |
| no header signal | 0.927 | -0.062 | -0.089 to -0.035 |
| no one-file-per-entity | 0.947 | -0.043 | -0.095 to +0.000 |
| no relation signal | 0.965 | -0.024 | -0.043 to -0.009 |
| no date rule | 0.967 | -0.023 | -0.046 to -0.006 |
| no magnitude rule | 0.983 | -0.006 | -0.015 to +0.000 |

### Human review

The mapper flags 21 of 329 mapped columns (6.4%) for a person to look at (low confidence, a close second choice, or decided by a tie-break rule). 0.0% of the 4 wrong predictions are among the flagged columns, and 1.3% of the unflagged columns are wrong.

### Normalization

Records found and field values equal to the truth, on the held-out tenants' first export. 'Oracle mapping' uses the true mapping, so it isolates the transform layer; 'predicted mapping' adds the mapper's mistakes.

| | Records found | Field accuracy | Rows loaded that should not be | Junk rows held back |
|---|---|---|---|---|
| oracle mapping | 100.0% | 99.3% | 0 | 100.0% |
| predicted mapping | 95.1% | 99.1% | 116 | 100.0% |

Wrong field values with the oracle mapping, by field: invoice.status 119, customer.country 63, payment.method 43. These are the limits of the transform layer: status and payment-method words and country names outside the vocabulary, and amounts in minor units (cents) in files with no header that says so.

### Schema drift

Each held-out tenant has a second export with 3 or 4 injected changes (renamed, added, dropped, retyped columns) and a clean re-export with new and changed rows but no schema change. Rename threshold 0.7, chosen on the tuning tenants.

| Config built from | Events found | Precision | Recall | Syncs decided right | Unsafe misses | False alarms on clean re-exports |
|---|---|---|---|---|---|---|
| oracle mapping | 68 | 1.000 | 0.985 | 100.0% | 0 of 17 | 0 events, 0 false blocks (20 syncs) |
| predicted mapping | 68 | 1.000 | 0.985 | 100.0% | 0 of 17 | 0 events, 0 false blocks (20 syncs) |

Detected by kind (predicted mapping): added 11/11, dropped_optional 6/6, dropped_required 6/6, renamed 13/13, retyped_format 17/17, retyped_incompatible 14/15.

In the 17 exports that should be blocked and were, loading them with the old config would have written 1,784 wrong or missing field values out of 18,475 (9.7%).

### Ingest guarantees

Run on the held-out tenants with the true mapping, so the ledger is tested rather than the mapper: 100 of 100 checks pass.

| Check | Passed | Scenario actually occurred |
|---|---|---|
| idempotent | 20 of 20 | 20 of 20 |
| incremental | 20 of 20 | 20 of 20 |
| dry run | 20 of 20 | 20 of 20 |
| crash and resume | 20 of 20 | 20 of 20 |
| rollback | 20 of 20 | 20 of 20 |
