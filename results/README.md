# Paper results

These files contain aggregate statistics for the MM-JailBench manuscript. Full results use the archived final GPT-5 judgments for both p1 and p2 of all sixteen evaluated models: 16,320 unique factor cells per model and 261,120 scored responses overall. The attack-success threshold is score ≥ 4.

| File | Contents |
|---|---|
| model_results.csv | Per-model Text/OCR/pooled ASR, CASR, mean scores, and domain-macro ASR |
| carrier.csv | Carrier-level counts, successes, mean scores, and ASR across all models |
| prompt.csv | Prompt-template aggregates across all models and carriers |
| visual.csv | Five semantic-image aggregates across all models and carriers |
| domain.csv | Domain-level aggregates across all models and carriers |
| lite_seed62_aggregate.csv | Full versus selected Lite agreement by carrier and pooled |
| lite_seed62_cells.csv | All 48 model-by-carrier Full/Lite comparisons supporting the 0.72-point mean absolute difference |
| provenance.json | Archived input checksums, row counts, and aggregate verification |

Union means pooled Text/OCR instances. A model's pooled ASR is not its equal-domain macro ASR. In the balanced full design, pooling is equivalent to averaging that model's Text and OCR ASRs before rounding.

The selected Lite split is seed62: 1,500 instances per model, 259 intents, all nine domains, all eighteen scenarios, and every prompt/image/carrier level. Full/Lite aggregates in lite_seed62_aggregate.csv pool sixteen models, so their row counts are sixteen times the per-model dataset sizes.

Archived judgments remain outside this repository. The published file checksums identify the source artifacts without exposing response text. [scripts/summarize_results.py](../scripts/summarize_results.py) reproduces per-model metrics from the corresponding judged CSV files; archived comparisons were checked before this release.

Results are historical measurements of the paper's configurations. Do not attribute them to a different checkpoint, newer API version, or different decoding budget without rerunning the benchmark.
