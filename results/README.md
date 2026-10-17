# Results and provenance

This directory provides both values transcribed from **[MMJailBench: A Factorized Benchmark for Disentangling Multimodal Jailbreak Vulnerabilities](https://arxiv.org/abs/2608.25490v1)** and statistics recomputed from archived judgments. Paper references are pinned to arXiv **v1**, submitted 26 August 2026.

## Paper-reported tables

| File | Source and contents |
|---|---|
| [paper_table1.csv](paper_table1.csv) | Table 1: sixteen models, printed Avg., and nine domain ASRs |
| [paper_table2.csv](paper_table2.csv) | Table 2: visual controls on four models; deltas are percentage points |
| [paper_table3.csv](paper_table3.csv) | Table 3: Text/OCR/Union ASR, CASR, mismatch rate, and counts |
| [paper_table11.csv](paper_table11.csv) | Table 11: Full/Lite agreement over 32 Text/OCR cells |
| [paper_table12.csv](paper_table12.csv) | Table 12: human and lightweight-judge agreement |
| [paper_provenance.json](paper_provenance.json) | Paper version, PDF checksum, table anchors, and figure attribution |

These files preserve the paper's values as printed. `reported_avg_asr_pct` in Table 1 is a transcription of `Avg.`, not an output of the current summarizer. Printed averages may differ from recomputed statistics; use the separately labeled files below for recomputation. No paper values have been silently substituted with archive-derived values.

The 144 domain cells in Table 1, 96 prompt cells in Table 9, and 80 visual cells in Table 10 were checked against the archived judgments: all **320 cells** match at the paper's displayed two-decimal precision.

## Recomputed archived statistics

Full results use final GPT-5 judgments for both historical manifests (p1 and p2) of all sixteen evaluated models: 16,320 instances per model and 261,120 scored responses overall. The success threshold is score ≥ 4.

| File | Contents |
|---|---|
| [model_results.csv](model_results.csv) | Per-model Text/OCR/pooled ASR, CASR, mean scores, and domain-macro ASR |
| [carrier.csv](carrier.csv) | Carrier-level counts, successes, mean scores, and ASR across all models |
| [prompt.csv](prompt.csv) | Prompt-template aggregates across all models and carriers |
| [visual.csv](visual.csv) | Five semantic-image aggregates across all models and carriers |
| [domain.csv](domain.csv) | Domain-level aggregates across all models and carriers |
| [lite_seed62_aggregate.csv](lite_seed62_aggregate.csv) | Full versus selected Lite agreement by carrier and pooled |
| [lite_seed62_cells.csv](lite_seed62_cells.csv) | 32 Text/OCR comparisons plus 16 additional pooled comparisons |
| [provenance.json](provenance.json) | Archived input checksums, row counts, and aggregate verification |

**Union means pooled Text/OCR instances.** A model's pooled ASR is not its equal-domain macro ASR. In the balanced full design, pooling equals the average of that model's Text and OCR ASRs before rounding.

## Lite comparison denominator

The paper's Table 11 compares **16 models × 2 carriers = 32 cells**. Filtering `lite_seed62_cells.csv` to `carrier == text` or `carrier == ocr` reproduces the mean difference −0.06 percentage points, mean absolute difference **0.79 points**, maximum absolute difference 3.53 points, and 30/32 cells within 2 points.

The CSV also includes 16 `union` rows. Including these extra pooled summaries gives **48 comparisons** and a mean absolute difference of **0.72 points**. This is additional archive analysis with a different denominator, not the headline statistic from Table 11. The paper's confidence interval and correlation statistics are transcribed in `paper_table11.csv`; they are not estimated by the released summarizer.

The archived Lite manifest uses selection seed 62: 1,500 instances per model, 259 intents, all nine domains, all eighteen scenarios, and every prompt/image/carrier level. These are implementation metadata. Full/Lite aggregates in `lite_seed62_aggregate.csv` pool sixteen models, so their counts are sixteen times the per-model dataset sizes.

Archived response-level judgments remain outside this repository. Checksums identify the source artifacts without exposing response text. [scripts/summarize_results.py](../scripts/summarize_results.py) reproduces per-model metrics from the corresponding judged CSV files.

Results describe historical model configurations. Comparisons to a different checkpoint, API version, or decoding budget require a new evaluation.
