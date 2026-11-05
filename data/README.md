# Dataset release

| Hugging Face viewer | Split | Rows |
|---|---|---:|
| [Full](https://huggingface.co/datasets/MMJail-Lab/MMJail-Bench/viewer/full/test) | `test` | 16,320 |
| [Lite-500](https://huggingface.co/datasets/MMJail-Lab/MMJail-Bench/viewer/lite/test) | `test` | 500 |

Both previews show Text and OCR examples with images. Text rows use `prompt`; OCR rows leave it empty. The viewer displays JPEG previews; the dataset shards retain the original image bytes.

Evaluation inputs are available on [Hugging Face](https://huggingface.co/datasets/MMJail-Lab/MMJail-Bench), with `full` (16,320 rows) and `lite` (500 rows) configurations and embedded original images. This GitHub directory contains aggregate taxonomy metadata; use the exporter in the [reproduction guide](../docs/reproduction.md) to download images and CLI-compatible manifests locally.

The [data-format guide](../docs/data_format.md) documents Hugging Face fields and CLI manifests. The [benchmark guide](../docs/benchmark.md) distinguishes Full, the current manifest-defined Lite-500, the historical paper Lite-1500, and the reviewer supplement.

Dataset assets use CC BY 4.0; code uses MIT. Versioned source and image checksums accompany the Hugging Face release. Raw responses, judge reasoning, and model weights are not part of the dataset.
