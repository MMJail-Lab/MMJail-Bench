# Reproduction guide

All commands below run from the repository root. Dataset manifests and semantic images will be distributed separately; the Hugging Face repository and access policy will be linked when available. This GitHub release does not include those dataset files or a local judge checkpoint.

## 1. Install and configure

Python 3.10–3.13 is supported by the package metadata. Install the core package with pip install -e ., and add the local-models extra for open-weight VLMs as shown in the [README](../README.md).

Copy configs/models.example.yaml to configs/models.yaml. The example covers Qwen2.5-VL, Qwen3-VL, Gemma3, InternVL3, GLM-4.1V, Ministral-3, LLaVA-OneVision, Step3-VL, and OpenAI-compatible APIs. Model files are downloaded independently and may require acceptance of their providers' terms.

For an API model, set both the credential variable named by api_key_env and the endpoint variable named by base_url_env. Set the selected model's path or its model-ID environment override to the endpoint's supported identifier. The template's environment-variable names contain no credentials.

For local inference, set model path/revision, device/device_map, and precision. The Qwen2.5-VL entry records a checkpoint revision. Other entries without revisions require you to record the snapshot actually used. Treat the template as the recorded configuration, rather than a promise that a current API alias matches the paper's evaluated version.

## 2. Understand the manifests

The historical filenames used in the experiments are:

| Purpose | p1 | p2 |
|---|---|---|
| Construction-stage manifest | data/dataset_full.jsonl | data/dataset_full_2.jsonl |
| Final inference manifest | data/dataset_full_ocr.jsonl | data/dataset_full_ocr_2.jsonl |
| Generated OCR directory | data/imgs_ocr/ | data/imgs_ocr_2/ |

Each part contains 8,160 records with unique sample IDs. p1 and p2 share the same IDs and swap carrier assignments. The reviewed CLI validates, resumes, and merges using sample_id within one part, so **keep parts separate through inference and judging**.

The build-data command reproduces the paper's **p2** carrier assignment. With the reviewer metadata it yields 180 records; with complete metadata and images it yields 8,160 p2 records. It does not construct a combined 16,320-row manifest or the fixed Lite selection.

~~~bash
python -m jailbreak_bench.cli build-data --task-info data/task_info.jsonl --task-labels data/task_label.jsonl --output data/dataset_base.rebuilt.jsonl
python -m jailbreak_bench.cli build-ocr --base-dataset data/dataset_base.rebuilt.jsonl --image-dir imgs_ocr_rebuilt --output-dataset data/dataset.rebuilt.jsonl
python -m jailbreak_bench.cli validate --dataset data/dataset.rebuilt.jsonl --data-root data --stage final
~~~

The OCR target directory must be empty and the output manifest must not exist. Original prompts remain in construction-stage records; final OCR records have empty prompt fields. See [data format](data_format.md).

For Lite, the historical 1,500-row manifest contains both carriers with repeated sample IDs. Partition it by carrier before passing it to the reviewed CLI, and use the corresponding original-prompt manifest for judging. Future dataset release instructions will describe the supplied manifests explicitly.

## 3. Run each part

Example for p1 using Qwen2.5-VL:

~~~bash
python -m jailbreak_bench.cli validate --dataset data/dataset_full_ocr.jsonl --data-root data --stage final
python -m jailbreak_bench.cli run-victim --config configs/models.yaml --model qwen2.5-vl-7b --dataset data/dataset_full_ocr.jsonl --data-root data --output outputs/victim_p1.csv --batch-size 4
python -m jailbreak_bench.cli run-judge --config configs/models.yaml --model gpt-5-judge --prompt-dataset data/dataset_full.jsonl --input outputs/victim_p1.csv --output outputs/judged_p1.csv
~~~

Repeat with the p2 manifests and distinct victim_p2.csv / judged_p2.csv output paths. Confirm that every referenced image exists before inference. The example commands require the corresponding dataset files, not only the code checkout.

The same configuration entries can be selected for API inference. Credentials are resolved by the selected adapter. API inference and judging may incur provider charges.

## 4. Resume or shard a single part

- Use --resume to skip sample IDs already in that part's output.
- Use --overwrite only when intentionally replacing an output. It is mutually exclusive with --resume.
- Use --limit for a smoke run, and label the result as partial.
- Use --rank and --world-size for index-based sharding within a part. Give each shard a different output path.

Merge rank shards **within the same part**:

~~~bash
python -m jailbreak_bench.cli merge --inputs outputs/p1_rank0.csv outputs/p1_rank1.csv --output outputs/victim_p1.csv
~~~

Do not use the default sample_id merge to combine p1 and p2. Aggregate judged parts using the script below instead. Failed rows may already be present in a resumed output; resolve those rows explicitly before aggregation.

## 5. Summarize

~~~bash
python scripts/summarize_results.py --inputs outputs/judged_p1.csv outputs/judged_p2.csv --output outputs/summary.json
~~~

Check that the pooled count is 16,320 and each carrier count is 8,160. Preserve the raw outputs privately alongside the exact data manifest hashes, configuration, model revision, timestamp, and environment. Compare ASR, CASR, and domain-macro ASR according to the [evaluation protocol](evaluation.md).

## Offline verification

The aggregation tests require only the Python standard library:

~~~bash
python -m unittest discover -s tests -v
~~~

These tests cover metric correctness and input validation. They do not validate GPU inference on every supported architecture or the availability of third-party API models.
