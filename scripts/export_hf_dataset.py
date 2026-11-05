# SPDX-License-Identifier: MIT
"""Export embedded images and carrier-safe manifests for the MMJailBench CLI."""
import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath



def output_path(root, relative):
    path = Path(relative)
    if path.is_absolute() or PureWindowsPath(relative).drive:
        raise ValueError('Image path must be relative')
    target = (root / path).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError('Image path escapes output directory')
    return target


def export_rows(rows, output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose an empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    handles = {carrier: (output / f'dataset_{carrier}.jsonl').open('w', encoding='utf-8')
               for carrier in ['text', 'ocr']}
    references = {}
    images = {}
    seen = set()
    counts = {'text': 0, 'ocr': 0}
    try:
        for row in rows:
            carrier = row['carrier_mode']
            key = (row['sample_id'], carrier)
            if carrier not in handles or key in seen:
                raise ValueError('Invalid or duplicated carrier key')
            seen.add(key)
            if row['prompt'] != (row['reference_prompt'] if carrier == 'text' else ''):
                raise ValueError('Victim prompt violates the carrier contract')
            data = row['image']['bytes']
            if not data:
                raise ValueError('Embedded original image bytes required')
            relative = row['image_path']
            image_digest = hashlib.sha256(data).hexdigest()
            target = output_path(output, relative)
            if relative not in images:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                images[relative] = image_digest
            elif images[relative] != image_digest:
                raise ValueError('Conflicting images share one archive path')
            sample = {'sample_id': row['sample_id'],
                'task_meta': {k: row[k] for k in ['task_id','task_label','category','tier2','severity']},
                'input_config': {k: row[k] for k in ['query_type','prompt','carrier_mode','image_type','image_path']}}
            handles[carrier].write(json.dumps(sample, ensure_ascii=False) + '\n')
            if row['sample_id'] not in references:
                reference = {**sample, 'input_config': {**sample['input_config'], 'prompt': row['reference_prompt']}}
                references[row['sample_id']] = reference
            elif references[row['sample_id']]['input_config']['prompt'] != row['reference_prompt']:
                raise ValueError('Conflicting original instructions')
            counts[carrier] += 1
        with (output / 'judge_prompts.jsonl').open('w', encoding='utf-8') as stream:
            for sid in sorted(references):
                stream.write(json.dumps(references[sid], ensure_ascii=False) + '\n')
    finally:
        for handle in handles.values():
            handle.close()
    return counts


def main():
    from datasets import Image, load_dataset
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', choices=['full', 'lite'], default='lite')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo', default='MMJail-Lab/MMJail-Bench')
    args = parser.parse_args()
    dataset = load_dataset(args.repo, args.config, split='test', streaming=True)
    dataset = dataset.cast_column('image', Image(decode=False))
    print(json.dumps(export_rows(dataset, args.output), indent=2))


if __name__ == '__main__':
    main()
