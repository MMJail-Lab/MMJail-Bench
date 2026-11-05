# Data format

Dataset files are JSON Lines with UTF-8 encoding. The schema is shared by the reviewed construction, OCR rendering, inference, and judging utilities. Actual prompts and images are distributed separately.

## Hugging Face release

The `full` and `lite` configurations contain 16,320 and 500 rows respectively, each with a `test` split. They use flat Parquet fields: `id`, `sample_id`, task metadata, factor labels, `prompt`, `reference_prompt`, `image_path`, `source_part`, and an embedded `image` feature.

`prompt` is the actual victim text input: the original instruction for Text and exactly `""` for OCR. `reference_prompt` is judge-only and must not be appended to OCR victim inputs. `image` is the actual semantic or stitched OCR image. `id` includes the carrier; the raw `sample_id` does not.

[scripts/export_hf_dataset.py](../scripts/export_hf_dataset.py) exports the flat data into the nested schema below, materializes original PNG bytes, and partitions final manifests by carrier. It also writes a unique-sample `judge_prompts.jsonl` with nonempty original requests. This reference file is for judging only, not construction or OCR rebuilding.

## Record fields

| Field | Type | Meaning |
|---|---|---|
| sample_id | string | Historical intent–prompt–image cell identifier; carrier is not encoded |
| task_meta.task_id | integer or numeric string | Original intent ID |
| task_meta.task_label | string | Intent label |
| task_meta.category | string | Level-1 harm domain |
| task_meta.tier2 | string | Level-2 scenario |
| task_meta.severity | string | Source severity label |
| input_config.query_type | string | One of the six prompt templates |
| input_config.carrier_mode | string | text or ocr |
| input_config.image_type | string | auth_doc, danger, id_card, role, or scenario |
| input_config.prompt | string | Original prompt in base records; empty for final OCR records |
| input_config.image_path | string | Image path relative to the selected data root |

The historical muse- prefix in sample_id is preserved to keep archived experiments and Lite selection reproducible. It is not a separate dataset or a new release identifier.

## Construction and inference stages

**Base records** contain the original prompt and semantic image path for both carrier assignments. The OCR renderer consumes these fields.

**Final Text records** retain the original prompt and semantic image. **Final OCR records** reference the stitched image and have an empty prompt string. Judging still needs the original prompt from a base manifest.

The schema validator checks required fields, carrier labels, stage-specific prompt rules, unique sample IDs within one CLI invocation, and image-path containment/existence under the data root.

## Identity and paired carriers

The stable paired key is **(sample_id, carrier_mode)**. The two historical full manifests have identical sample ID sets with complementary carrier assignments. The full benchmark has 8,160 distinct sample IDs and 16,320 distinct paired keys.

The reviewed inference CLI accepts one manifest with unique sample IDs at a time. Run p1/p2 or Lite carrier partitions separately. The public result summarizer supports paired keys when combining the judged outputs.

## Judged CSV files

The inference CSV contains sample metadata, prompt/image fields, and victim_response. Judging adds judge_prompt_used, judge_score, judge_reasoning, and is_mismatch.

The summarizer needs sample_id, carrier_mode, category, query_type, image_type, judge_score, and is_mismatch. It never copies original prompts, response text, or judge reasoning into its aggregate JSON report.

CSV fields may contain newlines and quoted text. Use a CSV parser rather than counting lines or splitting on commas. Failed generations or malformed judgments are incomplete evaluations and must be resolved before benchmark aggregation.
