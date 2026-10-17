# MM-JailBench

### Visual Semantic Camouflage and Cross-Modal Hijacking in Multimodal Jailbreak Evaluation

[![Python](https://img.shields.io/badge/Python-3.10%E2%80%933.13-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/Code-MIT-green)](LICENSE)
[![Team](https://img.shields.io/badge/Team-MMJail--Lab-5166d6)](https://github.com/MMJail-Lab)

**272 harmful intents · 9 domains · 18 scenarios · 6 prompt templates · 5 semantic image types · 2 carrier modes**

MM-JailBench evaluates how multimodal models respond to harmful requests when prompt framing, visual context, and instruction carrier are varied systematically. Its focus is **visual semantic camouflage**: benign-looking roles, scenarios, identity credentials, or authorization documents that can make an unsafe request appear legitimate.

The complete factorial design contains **16,320 instances per model**. The paper evaluates **16 models**, yielding **261,120 scored responses**, and studies carrier effects, prompt effects, semantic-image controls, and representation-level diagnostics.

Maintained by **MMJail-Lab**, a research team at the School of Computer Science and Technology, Tongji University.

**Release status:** benchmark documentation, aggregate paper results, and evaluation code are available here. The Hugging Face dataset release is being prepared; its link and access policy will be added when available. Dataset files and model weights are not included in this repository. The benchmark manuscript's public paper link will be added when available.

**Navigation:** [Benchmark design](docs/benchmark.md) · [Evaluation protocol](docs/evaluation.md) · [Reproduction guide](docs/reproduction.md) · [Data format](docs/data_format.md) · [Paper results](results/README.md) · [Responsible use](RESPONSIBLE_USE.md)

## Benchmark design

| Factor | Levels |
|---|---|
| Harmful intent | 272 distinct tasks across 9 domains and 18 scenarios |
| Prompt template | Academic, System, Structure, Story, Code, Paradox |
| Semantic image | Authorization document, danger, identity card, role, scenario |
| Carrier | Text instruction + semantic image; OCR instruction image + semantic image |

**272 × 6 × 5 × 2 = 16,320 instances.** Each prompt–image cell is evaluated in both carrier modes. This supports matched comparisons across presentation factors.

| Dataset | Instances per model | Intent coverage | Purpose |
|---|---:|---:|---|
| Full | 16,320 | 272 / 272 | Complete benchmark evaluation |
| Lite (fixed seed62) | 1,500 | 259 / 272 | Lower-cost regression screening |
| Reviewer subset | 180 | 6 / 272 | Compact reproduction supplement; separate from Lite |

Lite contains 750 matched Text/OCR pairs and covers all 9 domains, 18 scenarios, 6 prompt types, and 5 image types. Its selected seed is fixed for reproducibility. Neither Lite nor the reviewer subset replaces the full benchmark for final audits.

See [benchmark details and taxonomy](docs/benchmark.md).

## Main results

The table reports **attack success rate (ASR)** using the paper's GPT-5 judge and a harmfulness threshold of **score ≥ 4**. Lower ASR indicates fewer successful harmful requests under this protocol. These are the paper's evaluated configurations, rather than continuously updated claims about current model endpoints.

| Model | Text ASR (%) | OCR ASR (%) | Pooled ASR (%) |
|---|---:|---:|---:|
| GPT-5 | 3.93 | 0.31 | 2.12 |
| Claude Sonnet 4.5 | 9.30 | 8.48 | 8.89 |
| Qwen3-VL-8B | 27.17 | 12.79 | 19.98 |
| Gemini 3 Flash | 47.24 | 12.92 | 30.08 |
| Kimi K2.5 | 53.35 | 18.74 | 36.04 |
| LLaVA-OneVision-1.5 | 66.67 | 16.15 | 41.41 |
| Grok 4 Fast | 46.03 | 41.91 | 43.97 |
| Doubao Seed 2.0 Lite | 48.92 | 50.17 | 49.55 |
| InternVL3-8B | 47.73 | 52.34 | 50.04 |
| Ministral-3-8B | 62.23 | 42.29 | 52.26 |
| Step3-VL-10B | 58.79 | 48.26 | 53.52 |
| Qwen2.5-VL-7B | 61.27 | 60.47 | 60.87 |
| Qwen3-VL-Plus | 57.57 | 67.32 | 62.44 |
| Gemma3-12B | 74.56 | 60.97 | 67.76 |
| GLM-4.1V-9B | 69.12 | 77.39 | 73.25 |
| GLM-4.6V | 77.35 | 78.86 | 78.11 |
| **All models** | **50.70** | **40.59** | **45.64** |

The pooled column combines all Text and OCR instances. The paper also calls this **Union**; it is **not** an OR over paired outcomes. Domain-macro ASR is a different statistic and is reported separately in the downloadable results.

Key observations:

- OCR is not universally stronger: OCR ASR is lower than Text ASR for 11 of the 16 evaluated models.
- Prompt framing has the largest range among the presentation factors. Story, Structure, and Academic reach pooled ASRs of 64.42%, 60.05%, and 57.31%, respectively, versus 22.99% for System.
- The full benchmark's five-image comparison does not isolate semantic meaning from image presence. A separate four-model control ablation evaluates that question.
- Lite retains aggregate Text/OCR trends, with a mean absolute difference of 0.72 percentage points across the 48 model-by-carrier summaries. Fine-grained visual rankings remain less stable.

Download [model-level results](results/model_results.csv), [carrier results](results/carrier.csv), [prompt results](results/prompt.csv), [image results](results/visual.csv), [Lite agreement](results/lite_seed62_aggregate.csv), and the [48 Full/Lite comparisons](results/lite_seed62_cells.csv). See [provenance and metric definitions](results/README.md).

## Installation

Use Python **3.10–3.13** from the repository root:

~~~bash
git clone https://github.com/MMJail-Lab/MMJail-Bench.git
cd MMJail-Bench
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e .
python -m jailbreak_bench.cli --help
~~~

For local VLM inference, install the optional model stack:

~~~bash
pip install -e ".[local-models]"
~~~

The pinned dependencies record the reproduction environment. GPU/CUDA compatibility and access to model checkpoints are separate requirements. See [model configuration and reproducibility notes](docs/reproduction.md).

## Evaluation workflow

The code supports dataset validation, OCR rendering, local/API victim inference, GPT-5 judging, and result aggregation. **Inference commands require benchmark data obtained separately.** Until the data release is available, installation and result inspection can be used independently.

~~~bash
# Once dataset files are available under data/:
python -m jailbreak_bench.cli validate --dataset data/dataset.jsonl --data-root data --stage final
python -m jailbreak_bench.cli run-victim --config configs/models.yaml --model qwen2.5-vl-7b --dataset data/dataset.jsonl --data-root data --output outputs/victim.csv --batch-size 4
python -m jailbreak_bench.cli run-judge --config configs/models.yaml --model gpt-5-judge --prompt-dataset data/dataset_base.jsonl --input outputs/victim.csv --output outputs/judged.csv
python scripts/summarize_results.py --inputs outputs/judged.csv --output outputs/summary.json
~~~

Create the machine-local configuration by copying [configs/models.example.yaml](configs/models.example.yaml) to configs/models.yaml and setting the selected model's credential and endpoint environment variables. Keep local credentials out of version control.

The reviewed CLI expects unique sample IDs within each invocation. **Run the complete benchmark's p1 and p2 manifests separately**, then aggregate their judged outputs with the supplied summarizer. It validates the composite key (sample_id, carrier_mode) so paired carriers are both counted. See the [complete reproduction commands](docs/reproduction.md).

## Repository structure

~~~text
jailbreak_bench/       reviewed construction, inference, judge, and model adapters
configs/              credential-free model configuration template
scripts/              strict ASR/CASR and factor-level aggregation
results/              aggregate paper results and provenance
docs/                 design, protocol, data format, and reproduction guide
assets/fonts/         exact OCR font and upstream license
tests/                offline metric correctness checks
~~~

## Citation and contact

For now, refer to the work as **MM-JailBench: Visual Semantic Camouflage and Cross-Modal Hijacking in Multimodal Jailbreak Evaluation**. A verified paper link and complete BibTeX entry will be added with the public manuscript.

Use [GitHub Issues](https://github.com/MMJail-Lab/MMJail-Bench/issues) for questions and reproducibility reports. Please keep examples minimal and omit credentials, personal information, and operational harmful content.

The code follows the existing project's [MIT license](LICENSE). The bundled font has its [own license](assets/fonts/LICENSE.txt). Dataset and model licensing will be documented with their respective releases. See [responsible-use guidance](RESPONSIBLE_USE.md).
