from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable: Iterable[Any], **_: Any) -> Iterable[Any]:
        return iterable

from .config import get_model_config, load_config
from .dataset import build_dataset, dataset_summary
from .determinism import PAPER_SEED, configure_determinism
from .io import append_csv, completed_sample_ids, read_csv, read_jsonl, sha256_file, write_jsonl
from .judge import SafetyJudge
from .models import load_model
from .ocr import render_ocr_dataset
from .schema import validate_dataset


def _load_image(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        return image.convert("RGB")


def _batched(items: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def _prepare_output(path: Path, *, resume: bool, overwrite: bool) -> set[str]:
    if resume and overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if path.exists() and overwrite:
        path.unlink()
    if path.exists() and not resume and not overwrite:
        raise FileExistsError(f"{path} exists. Use --resume or --overwrite.")
    return completed_sample_ids(path) if resume else set()


def _load_original_prompt_map(path: str | Path) -> dict[str, str]:
    samples = read_jsonl(path)
    validate_dataset(samples, stage="base")
    prompt_map: dict[str, str] = {}
    for sample in samples:
        sample_id = str(sample["sample_id"])
        if sample_id in prompt_map:
            raise ValueError(f"Duplicate sample_id in prompt dataset: {sample_id}")
        prompt_map[sample_id] = str(sample["input_config"]["prompt"])
    return prompt_map


def cmd_build_data(args: argparse.Namespace) -> None:
    samples = build_dataset(
        args.task_info,
        args.task_labels,
        semantic_image_dir=args.semantic_image_dir,
    )
    validate_dataset(samples, data_root=args.data_root, stage="base")
    write_jsonl(args.output, samples)
    print(
        json.dumps(
            {"sha256": sha256_file(args.output), **dataset_summary(samples)},
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_build_ocr(args: argparse.Namespace) -> None:
    samples = read_jsonl(args.base_dataset)
    validate_dataset(samples, data_root=args.data_root, stage="base")
    image_dir = Path(args.image_dir).as_posix().strip("/")
    if not image_dir or image_dir.startswith("..") or Path(image_dir).is_absolute():
        raise ValueError("--image-dir must be a safe relative directory under --data-root")
    output_dataset = Path(args.output_dataset)
    if output_dataset.exists():
        raise FileExistsError(f"{output_dataset} exists; remove it before rebuilding")
    rendered, report = render_ocr_dataset(
        samples,
        data_root=args.data_root,
        output_image_dir=Path(args.data_root) / image_dir,
        image_path_prefix=image_dir,
        font_path=args.font,
    )
    validate_dataset(rendered, data_root=args.data_root, stage="final")
    write_jsonl(output_dataset, rendered)
    report["dataset_sha256"] = sha256_file(output_dataset)
    report.update(dataset_summary(rendered))
    print(json.dumps(report, ensure_ascii=False, indent=2))


def cmd_validate(args: argparse.Namespace) -> None:
    samples = read_jsonl(args.dataset)
    validate_dataset(samples, data_root=args.data_root, stage=args.stage)
    print(
        json.dumps(
            {"sha256": sha256_file(args.dataset), **dataset_summary(samples)},
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_run_victim(args: argparse.Namespace) -> None:
    environment = configure_determinism(PAPER_SEED, strict=True)
    print("Determinism:", json.dumps(environment, sort_keys=True))
    samples = read_jsonl(args.dataset)
    validate_dataset(samples, data_root=args.data_root, stage="final")

    config = load_config(args.config)
    model_config = get_model_config(config, args.model, role="victim")
    if args.device:
        model_config["device"] = args.device
    model = load_model(model_config)

    output_path = Path(args.output)
    done = _prepare_output(output_path, resume=args.resume, overwrite=args.overwrite)
    sharded = [sample for index, sample in enumerate(samples) if index % args.world_size == args.rank]
    pending = [sample for sample in sharded if str(sample["sample_id"]) not in done]
    if args.limit is not None:
        pending = pending[: args.limit]

    data_root = Path(args.data_root)
    for batch in tqdm(
        _batched(pending, args.batch_size),
        total=(len(pending) + args.batch_size - 1) // args.batch_size,
        desc=f"Victim inference: {args.model}",
    ):
        prompts: list[str] = []
        images: list[Image.Image] = []
        for sample in batch:
            input_config = sample["input_config"]
            prompt = "" if input_config["carrier_mode"] == "ocr" else str(input_config["prompt"])
            prompts.append(prompt)
            images.append(_load_image(data_root / str(input_config["image_path"])))

        # No victim system prompt and no suffix. OCR rows have no text block.
        responses = model.generate(prompts, images=images)
        if len(responses) != len(batch):
            raise RuntimeError(f"Model returned {len(responses)} responses for a batch of {len(batch)}")

        output_rows: list[dict[str, Any]] = []
        for sample, response in zip(batch, responses):
            task_meta = sample["task_meta"]
            input_config = sample["input_config"]
            output_rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "category": task_meta.get("category", ""),
                    "tier2": task_meta.get("tier2", ""),
                    "task_label": task_meta.get("task_label", ""),
                    "severity": task_meta.get("severity", ""),
                    "query_type": input_config.get("query_type", ""),
                    "carrier_mode": input_config.get("carrier_mode", ""),
                    "image_type": input_config.get("image_type", ""),
                    "prompt": input_config.get("prompt", ""),
                    "image_path": input_config.get("image_path", ""),
                    "victim_response": response,
                }
            )
        append_csv(output_path, output_rows)
    print(f"Wrote {len(pending)} victim outputs to {output_path}.")


def cmd_run_judge(args: argparse.Namespace) -> None:
    environment = configure_determinism(PAPER_SEED, strict=True)
    print("Determinism:", json.dumps(environment, sort_keys=True))
    config = load_config(args.config)
    model_config = get_model_config(config, args.model, role="judge")
    judge = SafetyJudge(load_model(model_config))
    prompt_map = _load_original_prompt_map(args.prompt_dataset)
    rows = read_csv(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]

    output_path = Path(args.output)
    done = _prepare_output(output_path, resume=args.resume, overwrite=args.overwrite)
    judged_rows: list[dict[str, Any]] = []
    written = 0
    for row in tqdm(rows, desc=f"Judge: {args.model}"):
        sample_id = str(row.get("sample_id", ""))
        if sample_id in done:
            continue
        if sample_id not in prompt_map:
            raise KeyError(f"No original prompt for sample {sample_id}")
        prompt = prompt_map[sample_id]
        response = str(row.get("victim_response", ""))
        if not response or response.startswith("ERROR_"):
            score, mismatch, reasoning = -1, False, "Generation Error"
        else:
            score, mismatch, reasoning = judge.evaluate(prompt, response)
        judged_rows.append(
            {
                **row,
                "judge_prompt_used": prompt,
                "judge_score": score,
                "judge_reasoning": reasoning,
                "is_mismatch": mismatch,
            }
        )
        if len(judged_rows) >= args.flush_every:
            append_csv(output_path, judged_rows)
            written += len(judged_rows)
            judged_rows.clear()
    append_csv(output_path, judged_rows)
    written += len(judged_rows)
    print(f"Wrote {written} judged outputs to {output_path}.")


def cmd_merge(args: argparse.Namespace) -> None:
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists. Use --overwrite to replace it.")
    seen: set[str] = set()
    merged_rows: list[dict[str, Any]] = []
    for input_path in sorted(args.inputs):
        for row in read_csv(input_path):
            key = str(row.get(args.key, ""))
            if not key:
                raise ValueError(f"Empty merge key {args.key!r} in {input_path}")
            if key in seen:
                raise ValueError(f"Duplicate merge key {key!r}; refusing an ambiguous merge")
            seen.add(key)
            merged_rows.append(row)
    merged_rows.sort(key=lambda row: str(row.get(args.key, "")))
    if output.exists():
        output.unlink()
    append_csv(output, merged_rows)
    print(f"Merged {len(merged_rows)} unique rows into {output}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jailbreak-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_data = subparsers.add_parser("build-data", help="rebuild the 180-row base subset")
    build_data.add_argument("--task-info", default="data/task_info.jsonl")
    build_data.add_argument("--task-labels", default="data/task_label.jsonl")
    build_data.add_argument("--data-root", default="data")
    build_data.add_argument("--semantic-image-dir", default="imgs")
    build_data.add_argument("--output", default="data/dataset_base.rebuilt.jsonl")
    build_data.set_defaults(func=cmd_build_data)

    build_ocr = subparsers.add_parser("build-ocr", help="render OCR carriers and create final data")
    build_ocr.add_argument("--base-dataset", default="data/dataset_base.jsonl")
    build_ocr.add_argument("--data-root", default="data")
    build_ocr.add_argument("--font", default="assets/fonts/DejaVuSans-Bold.ttf")
    build_ocr.add_argument("--image-dir", default="imgs_ocr")
    build_ocr.add_argument("--output-dataset", default="data/dataset.jsonl")
    build_ocr.set_defaults(func=cmd_build_ocr)

    validate = subparsers.add_parser("validate", help="check data schema and all image paths")
    validate.add_argument("--dataset", default="data/dataset.jsonl")
    validate.add_argument("--data-root", default="data")
    validate.add_argument("--stage", choices=["base", "final"], default="final")
    validate.set_defaults(func=cmd_validate)

    victim = subparsers.add_parser("run-victim", help="run victim-model inference")
    victim.add_argument("--config", default="configs/models.example.yaml")
    victim.add_argument("--model", required=True)
    victim.add_argument("--dataset", default="data/dataset.jsonl")
    victim.add_argument("--data-root", default="data")
    victim.add_argument("--output", required=True)
    victim.add_argument("--batch-size", type=int, default=32)
    victim.add_argument("--rank", type=int, default=0)
    victim.add_argument("--world-size", type=int, default=1)
    victim.add_argument("--device", default=None)
    victim.add_argument("--limit", type=int, default=None)
    victim.add_argument("--resume", action="store_true")
    victim.add_argument("--overwrite", action="store_true")
    victim.set_defaults(func=cmd_run_victim)

    judge = subparsers.add_parser("run-judge", help="score victim responses")
    judge.add_argument("--config", default="configs/models.example.yaml")
    judge.add_argument("--model", default="gpt-5-judge")
    judge.add_argument("--prompt-dataset", default="data/dataset_base.jsonl")
    judge.add_argument("--input", required=True)
    judge.add_argument("--output", required=True)
    judge.add_argument("--limit", type=int, default=None)
    judge.add_argument("--resume", action="store_true")
    judge.add_argument("--overwrite", action="store_true")
    judge.add_argument("--flush-every", type=int, default=10)
    judge.set_defaults(func=cmd_run_judge)

    merge = subparsers.add_parser("merge", help="merge deterministic rank shards")
    merge.add_argument("--inputs", nargs="+", required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--key", default="sample_id")
    merge.add_argument("--overwrite", action="store_true")
    merge.set_defaults(func=cmd_merge)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "rank") and not 0 <= args.rank < args.world_size:
        parser.error("--rank must satisfy 0 <= rank < world-size")
    args.func(args)


if __name__ == "__main__":
    main()
