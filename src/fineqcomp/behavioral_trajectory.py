"""Measure task behavior and serialized adapter size along fixed SFT runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from fineqcomp.adapters import adapter_tensors, apply_adapter_tensors
from fineqcomp.artifacts import claim_run, read_json, write_json
from fineqcomp.codec import (
    decode_adapter_tensor_map, encode_tensor_map, pad_lora_rank, truncate_lora_rank,
)
from fineqcomp.config import RunSpec
from fineqcomp.data import Example, _convert_gsm8k, read_jsonl, restore_aligned_rationales
from fineqcomp.evaluation import evaluate_natural, write_predictions
from fineqcomp.modeling import ModelSession
from fineqcomp.training import causal_nll, train_adapter


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_adapter(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(tensors, temporary)
    temporary.replace(path)


def intervention(tensors: dict[str, torch.Tensor], kind: str, rank: int,
                 seed: int) -> dict[str, torch.Tensor]:
    """Reduce rank through SVD or a random subspace of the same learned update.

    Both factors' discarded directions are zero, so those directions stay zero
    under Adam after its moments are cleared. The adapter scaling stays fixed.
    """
    if kind == 'restart':
        return tensors
    full_rank = max(t.shape[0] for n, t in tensors.items() if '.lora_A.' in n)
    if not 0 < rank < full_rank:
        raise ValueError('intervention rank must be below the trained rank')
    if kind == 'truncate':
        return pad_lora_rank(truncate_lora_rank(tensors, rank), full_rank)
    if kind != 'random':
        raise ValueError(f'unknown intervention {kind}')
    balanced = truncate_lora_rank(tensors, full_rank)
    reduced = {}
    generator = torch.Generator().manual_seed(seed)
    for a_name in sorted(n for n in balanced if '.lora_A.' in n):
        b_name = a_name.replace('.lora_A.', '.lora_B.')
        a, b = balanced[a_name], balanced[b_name]
        q = torch.linalg.qr(torch.randn(full_rank, rank, generator=generator)).Q
        reduced[a_name], reduced[b_name] = q.T @ a, b @ q
    return pad_lora_rank(reduced, full_rank)


def select_files(rows: list[dict], base: float, reference: float,
                 retentions: list[float], minimum_change: float) -> list[dict]:
    """Choose actual tested files at one fixed signed reference change."""
    change = reference - base
    selections = []
    for retention in retentions:
        threshold = base + retention * change
        valid = abs(change) >= minimum_change
        candidates = [r for r in rows if valid and (
            r['accuracy'] >= threshold if change > 0 else r['accuracy'] <= threshold
        )]
        winner = min(candidates, key=lambda r: (r['file_bits'], r['key'])) if candidates else None
        selections.append({
            'retention': retention, 'threshold': threshold,
            'direction': 'acquisition' if change > 0 else 'damage',
            'status': 'selected' if winner else ('unreachable' if valid else 'reference_too_small'),
            'key': winner['key'] if winner else None,
            'file_bits': winner['file_bits'] if winner else None,
        })
    return selections


def study_data(config: dict, seed: int, prepared: Path) -> dict[str, list[Example]]:
    root = prepared / 'natural' / 'cot_math_permuted' / f'seed{seed}'
    permuted = read_jsonl(root / 'train.jsonl')
    aligned = restore_aligned_rationales(permuted, 'The answer is:')
    calibration = restore_aligned_rationales(
        read_jsonl(root / 'calibration.jsonl'), 'The answer is:')
    training_prompts = {x.prompt for x in aligned}
    calibration = [x for x in calibration if x.prompt not in training_prompts]
    if len(calibration) < config['calibration_rows']:
        raise ValueError('too few disjoint calibration prompts')
    calibration = [replace(x, metadata={**x.metadata, 'evaluator': 'gsm8k'})
                   for x in calibration[:config['calibration_rows']]]
    test = read_jsonl(root / 'test.jsonl')[:config['gsm8k_test_rows']]
    source = Path(config['symbolic_source'])
    if sha256(source) != config['symbolic_sha256']:
        raise ValueError('GSM-Symbolic source differs from the pinned release')
    symbolic = []
    for row in map(json.loads, source.read_text().splitlines()):
        if row['instance'] in config['symbolic_instances']:
            example = _convert_gsm8k([row], 'symbolic')[0]
            symbolic.append(replace(example,
                example_id=f"symbolic-{row['original_id']}-{row['instance']}",
                metadata={**example.metadata, 'template': row['original_id'],
                          'instance': row['instance']}))
    if len(symbolic) != 100 * len(config['symbolic_instances']):
        raise ValueError('GSM-Symbolic must cover all 100 templates')
    if {x.prompt for x in aligned} & {x.prompt for x in calibration}:
        raise ValueError('training and calibration prompts overlap')
    return {'aligned': aligned, 'permuted': permuted,
            'calibration': calibration, 'gsm8k': test, 'symbolic': symbolic}


def prepare(config: dict, out: Path, runs: Path, prepared: Path) -> dict:
    source = read_json(config['source_lock'])
    specs = []
    for arm in config['arms']:
        groups = source['arms'][arm]['runs'] if arm == 'aligned' else [source['arms'][arm]]
        for group in groups:
            for run_id in group['run_ids']:
                path = runs / run_id / 'config.json'
                run = RunSpec.from_dict(read_json(path))
                if run.model.key in config['models'] and run.seed in config['seeds']:
                    specs.append((arm, run))
    cells = []
    for arm, run in sorted(specs, key=lambda x: (x[1].model.key, x[1].seed, x[0])):
        kinds = config.get('kinds', ['existing', 'trajectory'] + (
            config['interventions'] if arm == 'aligned' else []))
        for kind in kinds:
            cells.append({'id': len(cells), 'arm': arm, 'kind': kind, 'run': run.to_dict(),
                          'slug': f'{run.model.key}/{arm}/seed{run.seed}/{kind}'})
    if len(specs) != len(config['models']) * len(config['seeds']) * len(config['arms']):
        raise ValueError('source lock does not cover the requested panel')
    inputs = [Path(config['source_lock']), Path(config['symbolic_source'])]
    inputs.extend(Path(config[k]) for k in ('field_config', 'field_campaign'))
    for seed in config['seeds']:
        study_data(config, seed, prepared)
        inputs.extend((prepared / 'natural' / 'cot_math_permuted' / f'seed{seed}').glob('*.jsonl'))
    inputs.extend(runs / run.run_id / 'config.json' for _, run in specs)
    record = {'config': config, 'cells': cells,
              'inputs': {str(p): sha256(p) for p in inputs},
              'implementation': {str(p): sha256(p) for p in Path('src/fineqcomp').glob('*.py')}}
    previous = read_json(out / 'lock.json')
    if previous is not None and previous != record:
        raise ValueError('study changed: use a new output directory')
    write_json(out / 'lock.json', record)
    return record


def train_cell(session: ModelSession, run: RunSpec, cell: dict, data: dict,
               config: dict, root: Path) -> None:
    if read_json(root / 'training.json') is not None:
        return
    steps = set(config['checkpoints']) | {config['updates']}
    branch = cell['kind'] in config['interventions']
    # All runs use the same full horizon, batch and data order. Saving never
    # shortens the schedule. Re-execution starts from the same seeded init.
    spec = replace(run.training, epochs=2, max_updates=config['updates'],
                   restore_best=False)
    if run.adapter.dropout:
        raise ValueError('paired trajectories require zero adapter dropout')

    def checkpoint(step: int, optimizer: torch.optim.Optimizer) -> None:
        if step in steps:
            save_adapter(root / f'step{step}.pt', adapter_tensors(session.model, 'full_lora'))
            write_json(root / f'step{step}.json', {'step': step,
                'learning_rate': optimizer.param_groups[0]['lr']})
        if branch and step == config['intervention_step']:
            tensors = adapter_tensors(session.model, 'full_lora')
            save_adapter(root / 'before_intervention.pt', tensors)
            apply_adapter_tensors(session.model, intervention(
                tensors, cell['kind'], config['intervention_rank'], run.seed))
            # Same reset for the ordinary continuation and both rank controls.
            optimizer.state.clear()
            save_adapter(root / 'after_intervention.pt', adapter_tensors(session.model, 'full_lora'))

    metrics = train_adapter(session.model, session.tokenizer, data[cell['arm']],
        data['calibration'], run.model, spec, run.seed, root / 'training.jsonl',
        on_update=checkpoint)
    if metrics['optimizer_updates'] != config['updates']:
        raise ValueError('training stopped before the locked horizon')
    write_json(root / 'training.json', metrics)


def score(session: ModelSession, run: RunSpec, examples: list[Example],
          config: dict, path: Path) -> dict:
    cached = read_json(path)
    if cached is not None:
        return cached
    metrics, predictions = evaluate_natural(session.model, session.tokenizer,
        examples, run.model, 'gsm8k', config['generation_batch_size'],
        max_new_tokens=config['max_new_tokens'], generation_seed=run.seed)
    for prediction, example in zip(predictions, examples, strict=True):
        prediction['cluster'] = example.metadata.get('template', example.example_id)
    write_predictions(path.with_suffix('.jsonl'), predictions)
    write_json(path, metrics)
    print(f'{path}: accuracy={metrics["exact_match"]:.4f}', flush=True)
    return metrics


def codec_candidates(tensors: dict, config: dict, seed: int):
    """Existing rank/precision grid, with optional controls for recovery."""
    for rank in config['ranks']:
        reduced = truncate_lora_rank(tensors, rank)
        for bits in config['bits']:
            yield f'r{rank}_b{bits}', reduced, bits
    if config.get('scale_controls'):
        full_rank = max(t.shape[0] for n, t in tensors.items() if '.lora_A.' in n)
        head = truncate_lora_rank(tensors, 1)
        # Use the transmitted scales and the actual decoder for the norm target.
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'binary.fqcb'
            encode_tensor_map(head, path, 1)
            _, binary = decode_adapter_tensor_map(path)
        for label, source in [('head', head), ('full', truncate_lora_rank(tensors, full_rank))]:
            matched = {}
            for a_name in sorted(n for n in source if '.lora_A.' in n):
                b_name = a_name.replace('.lora_A.', '.lora_B.')
                a, b = source[a_name], source[b_name]
                target = binary[a_name].norm() * binary[b_name].norm()
                # ||BA||_F from small Gram matrices avoids a dense weight update.
                norm = ((b.T @ b) * (a @ a.T)).sum().clamp_min(0).sqrt()
                matched[a_name], matched[b_name] = a, b * (target / norm.clamp_min(1e-30))
            yield f'binary_norm_{label}_b16', matched, 16
            for scale in config['scale_sweep']:
                scaled = {n: t * scale if '.lora_B.' in n else t for n, t in source.items()}
                yield f'scale_{label}_{scale:g}_b16', scaled, 16
    if not config.get('spectral_controls'):
        return
    full_rank = max(t.shape[0] for n, t in tensors.items() if '.lora_A.' in n)
    balanced = truncate_lora_rank(tensors, full_rank)
    tail, scaled = {}, {}
    for a_name in sorted(n for n in balanced if '.lora_A.' in n):
        b_name = a_name.replace('.lora_A.', '.lora_B.')
        a, b = balanced[a_name], balanced[b_name]
        tail[a_name], tail[b_name] = a[1:].contiguous(), b[:, 1:].contiguous()
        # Balanced factors have squared row norms equal to singular values.
        # Match the rank-one update's Frobenius norm in every projection.
        singular = a.square().sum(1)
        ratio = singular[0] / singular.norm().clamp_min(1e-30)
        scaled[a_name], scaled[b_name] = a, b * ratio
    yield f'tail_r{full_rank - 1}_b16', tail, 16
    random = intervention(tensors, 'random', 1, seed)
    yield 'random_r1_b16', truncate_lora_rank(random, 1), 16
    yield f'norm_full_r{full_rank}_b16', scaled, 16


def evaluate_checkpoint(session: ModelSession, run: RunSpec, tensors: dict,
                        reference: dict, baseline: dict, data: dict, config: dict,
                        root: Path) -> None:
    if (root / 'selection.json').exists():
        # A selection is written before testing; a restart must still finish tests.
        selected = read_json(root / 'selection.json')
    else:
        rows = []
        for key, reduced, bits in codec_candidates(tensors, config, run.seed):
            path = root / f'{key}.fqcb'
            path.parent.mkdir(parents=True, exist_ok=True)
            storage = encode_tensor_map(reduced, path, bits)
            _, decoded = decode_adapter_tensor_map(path)
            apply_adapter_tensors(session.model, pad_lora_rank(decoded, run.adapter.rank))
            measured = score(session, run, data['calibration'], config, root / f'{key}.json')
            rank = max(t.shape[0] for n, t in reduced.items() if '.lora_A.' in n)
            rows.append({'key': key, 'rank': rank, 'bits': bits,
                         'file_bits': storage['file_bits'], 'accuracy': measured['exact_match']})
        write_json(root / 'candidates.json', rows)
        selected = select_files(rows, baseline['calibration']['exact_match'],
            reference['calibration']['exact_match'], config['retentions'],
            config['minimum_reference_change'])
        write_json(root / 'selection.json', selected)
    keys = {row['key'] for row in selected if row['key']}
    if config.get('test_all_candidates'):
        keys = {row['key'] for row in read_json(root / 'candidates.json')}
    for key in sorted(keys):
        _, decoded = decode_adapter_tensor_map(root / f'{key}.fqcb')
        apply_adapter_tensors(session.model, pad_lora_rank(decoded, run.adapter.rank))
        for split in ('gsm8k', 'symbolic'):
            score(session, run, data[split], config, root / f'{key}_{split}.json')
    apply_adapter_tensors(session.model, tensors)
    for split in ('calibration', 'gsm8k', 'symbolic'):
        score(session, run, data[split], config, root / f'raw_{split}.json')
    nll = causal_nll(session.model, session.tokenizer, data['calibration'], run.model,
                     run.training.max_length, run.training.micro_batch_size)
    write_json(root / 'raw_nll.json', nll)
    # Keep selected, decoded files as the deliverable. Other candidates can be
    # reconstructed from the checkpoint; their measured byte counts stay recorded.
    for path in root.glob('*.fqcb'):
        if path.stem not in keys:
            path.unlink()
    write_json(root / 'complete.json', {'complete': True})


def run_cell(cell: dict, config: dict, out: Path, runs: Path, prepared: Path,
             phase: str) -> None:
    root = out / cell['slug']
    run = RunSpec.from_dict(cell['run'])
    with claim_run(root) as acquired:
        if not acquired:
            raise RuntimeError(f'cell already running: {root}')
        if phase == 'train' and cell['kind'] == 'existing':
            return
        data = study_data(config, run.seed, prepared)
        if config.get('smoke'):
            data = {key: value[:32 if key in ('aligned', 'permuted') else 4]
                    for key, value in data.items()}
        session = ModelSession.load(run.model)
        try:
            session.attach(run.adapter, run.seed)
            if phase == 'train':
                train_cell(session, run, cell, data, config, root)
                return
            if cell['kind'] == 'existing':
                checkpoints = [('final', runs / run.run_id / 'raw_channel.pt')]
            else:
                if not (root / 'training.json').exists():
                    raise ValueError('trajectory training has not completed')
                steps = config['checkpoints'] if cell['kind'] == 'trajectory' else [config['updates']]
                checkpoints = [(f'step{s}', root / f'step{s}.pt') for s in steps]
            reference_path = checkpoints[-1][1]
            if cell['kind'] in config['interventions']:
                reference_path = root.parent / 'restart' / f'step{config["updates"]}.pt'
            reference_tensors = torch.load(reference_path, map_location='cpu', weights_only=True)
            baseline, reference = {}, {}
            for label, tensors in [('base', adapter_tensors(session.model, 'full_lora')),
                                   ('reference', reference_tensors)]:
                apply_adapter_tensors(session.model, tensors)
                metrics = {split: score(session, run, data[split], config,
                    root / f'{label}_{split}.json') for split in ('calibration', 'gsm8k', 'symbolic')}
                if label == 'base':
                    baseline = metrics
                else:
                    reference = metrics
            for label, path in checkpoints:
                destination = root / label
                if not (destination / 'complete.json').exists():
                    evaluate_checkpoint(session, run,
                        torch.load(path, map_location='cpu', weights_only=True),
                        reference, baseline, data, config, destination)
            write_json(root / 'complete.json', {'complete': True})
        finally:
            session.unload()


def matched_field(config: dict, out: Path, index: int) -> None:
    """Same post-prefix target, inputs, and exposure counts on each receiver."""
    from fineqcomp import program_selection as ps
    from fineqcomp.campaign import _adapter
    from fineqcomp.config import ModelSpec, TrainingSpec, load_campaign

    pairs = [(model, seed) for model in config['field_models'] for seed in config['seeds']]
    if not 0 <= index < len(pairs):
        raise ValueError('field cell is outside the model-seed panel')
    model, seed = pairs[index]
    root = out / 'field' / model / f'seed{seed}'
    ps_config = ps.load_config(Path(config['field_config']))
    campaign = load_campaign(config['field_campaign'])
    family = ps.FAMILIES['field']
    split = ps.split_pools(family, ps.build_pools(family), seed, ps_config)
    # Candidate indices and labels are identical across models, fixed before
    # measuring any receiver. No selection uses the post-training test set.
    candidates = split.candidates[:2]
    contract = {'target': config['field_target'], 'incumbent': config['field_incumbent'],
                'candidates': candidates, 'exposures': config['field_exposures'],
                'prefix_updates': 256, 'continuation_updates': 32, 'rows': 512}
    write_json(root / 'contract.json', contract)
    with claim_run(root) as acquired:
        if not acquired:
            raise RuntimeError('field cell is already running')
        if (root / 'complete.json').exists():
            return
        session = ModelSession.load(ModelSpec(key=model, **campaign['models'][model]))
        try:
            session.attach(_adapter(ps_config['adapter'], campaign), seed)
            spec = TrainingSpec(**ps_config['training'])
            prefix = ps.training_examples(family, split.base, config['field_target'])
            validation = ps.training_examples(family, split.ambiguous_eval[:64], config['field_target'])
            prefix_path = root / 'prefix.pt'
            if prefix_path.exists():
                apply_adapter_tensors(session.model, torch.load(prefix_path, weights_only=True))
            else:
                metrics = train_adapter(session.model, session.tokenizer, prefix, validation,
                    session.spec, replace(spec, epochs=32, max_updates=256), seed,
                    root / 'prefix_training.jsonl')
                if metrics['optimizer_updates'] != 256:
                    raise ValueError('field prefix did not reach 256 updates')
                save_adapter(prefix_path, adapter_tensors(session.model, 'full_lora'))
            state = adapter_tensors(session.model, 'full_lora')
            diagnostic = ps.program_mass(session, family, split.diagnostic, ps_config)
            fit = ps.ambiguous_accuracy(session, family, split.ambiguous_eval, ps_config)
            agreement = ps.program_agreement(session, family, split.probe, ps_config)
            write_json(root / 'prefix_diagnostic.json', {'mass': diagnostic['mass'],
                'agreement': agreement['agreement'], 'training_region_accuracy': fit,
                'incumbent_confirmed': max(agreement['agreement'], key=agreement['agreement'].get)
                                       == config['field_incumbent']})
            # Never screen a receiver away: a failed incumbent check limits the
            # interpretation but does not erase its continuation outcomes.
            arms = [('ordinary', None, 0)] + [
                (f'input{i}_exposures{n}', row, n) for i, row in enumerate(candidates)
                for n in config['field_exposures']]
            for name, row, count in arms:
                if (root / name / 'result.json').exists():
                    continue
                apply_adapter_tensors(session.model, state)
                inputs = list(split.extra[:512])
                for position in range(count):
                    inputs[position] = row
                examples = ps.training_examples(family, inputs, config['field_target'])
                metrics = train_adapter(session.model, session.tokenizer, examples, validation,
                    session.spec, replace(spec, epochs=1, max_updates=32, warmup_ratio=0.0),
                    seed, root / name / 'training.jsonl')
                if metrics['train_examples'] != 512 or metrics['optimizer_updates'] != 32:
                    raise ValueError('tokenization changed the locked exposure count')
                result = ps.program_agreement(session, family, split.probe, ps_config)
                write_json(root / name / 'result.json', {
                    'agreement': result['agreement'], 'unmatched_fraction': result['unmatched_fraction'],
                    'exposures': count, 'training': metrics,
                    'training_region_accuracy': ps.ambiguous_accuracy(session, family, split.ambiguous_eval, ps_config)})
                write_json(root / name / 'predictions.json', result['rows'])
            write_json(root / 'complete.json', {'complete': True})
        finally:
            session.unload()


def paired_interval(left: Path, right: Path, draws: int, seed: int) -> list[float]:
    """Paired accuracy difference, clustered by template for GSM-Symbolic."""
    a = {r['example_id']: r for r in map(json.loads, left.read_text().splitlines())}
    b = {r['example_id']: r for r in map(json.loads, right.read_text().splitlines())}
    if a.keys() != b.keys() or not a:
        raise ValueError('prediction IDs must match exactly')
    groups: dict[Any, list[float]] = {}
    for key, row in a.items():
        groups.setdefault(row['cluster'], []).append(float(row['correct']) - float(b[key]['correct']))
    means = np.array([np.mean(group) for group in groups.values()])
    rng = np.random.default_rng(seed)
    # Bound memory independently of the number of bootstrap draws.
    samples = np.concatenate([rng.choice(means, (min(100, draws - i), len(means))).mean(1)
                              for i in range(0, draws, 100)])
    return np.quantile(samples, [0.025, 0.975]).tolist()


def report(lock: dict, out: Path) -> dict:
    rows, missing, candidate_results = [], [], []
    config = lock['config']
    for cell in lock['cells']:
        root = out / cell['slug']
        if not (root / 'complete.json').exists():
            missing.append(cell['slug'])
            continue
        for selection in sorted(root.glob('*/selection.json')):
            if config.get('test_all_candidates'):
                for candidate in read_json(selection.parent / 'candidates.json'):
                    for split in ('gsm8k', 'symbolic'):
                        result = selection.parent / f'{candidate["key"]}_{split}.json'
                        base = read_json(root / f'base_{split}.json')['exact_match']
                        ref = read_json(root / f'reference_{split}.json')['exact_match']
                        accuracy = read_json(result)['exact_match']
                        candidate_results.append({**candidate, 'calibration_accuracy': candidate['accuracy'],
                            'cell': cell['slug'], 'split': split, 'accuracy': accuracy,
                            'base_accuracy': base, 'reference_accuracy': ref,
                            'change_from_base': accuracy - base, 'change_from_raw': accuracy - ref,
                            'change_from_raw_ci95': paired_interval(result.with_suffix('.jsonl'),
                                root / f'reference_{split}.jsonl', config['bootstrap_draws'], config['analysis_seed'])})
            for chosen in read_json(selection):
                for split in ('gsm8k', 'symbolic'):
                    base = read_json(root / f'base_{split}.json')['exact_match']
                    ref = read_json(root / f'reference_{split}.json')['exact_match']
                    row = {**chosen, 'cell': cell['slug'], 'checkpoint': selection.parent.name,
                           'split': split, 'base_accuracy': base, 'reference_accuracy': ref}
                    if chosen['key']:
                        result = selection.parent / f'{chosen["key"]}_{split}.json'
                        accuracy = read_json(result)['exact_match']
                        row['accuracy'] = accuracy
                        row['change_from_base'] = accuracy - base
                        change = ref - base
                        same_direction = (change > 0) == (chosen['direction'] == 'acquisition')
                        row['test_threshold_met'] = same_direction and abs(change) >= config['minimum_reference_change'] and (
                            accuracy - base >= chosen['retention'] * change if change > 0
                            else accuracy - base <= chosen['retention'] * change)
                        row['paired_change_ci95'] = paired_interval(result.with_suffix('.jsonl'),
                            root / f'base_{split}.jsonl', config['bootstrap_draws'], config['analysis_seed'])
                    rows.append(row)
    comparisons = []
    for cell in lock['cells']:
        root = out / cell['slug']
        if cell['kind'] == 'trajectory' and (root / 'complete.json').exists():
            for split in ('gsm8k', 'symbolic'):
                early, late = root / 'step512', root / f'step{config["updates"]}'
                if not (early / f'raw_{split}.jsonl').exists():
                    continue
                interval = paired_interval(late / f'raw_{split}.jsonl', early / f'raw_{split}.jsonl',
                                           config['bootstrap_draws'], config['analysis_seed'])
                budgets = [next((r for r in rows if r['cell'] == cell['slug']
                    and r['checkpoint'] == point.name and r['split'] == split
                    and r['retention'] == 0.9), None) for point in (early, late)]
                ratio = budgets[1]['file_bits'] / budgets[0]['file_bits'] if all(
                    b and b.get('test_threshold_met') for b in budgets) else None
                comparisons.append({'cell': cell['slug'], 'split': split, 'kind': 'late_concentration',
                    'raw_late_minus_early_ci95': interval, 'late_over_early_bytes': ratio,
                    'passed': ratio is not None and ratio <= config['storage_ratio']
                    and interval[0] >= -config['equivalence_margin']
                    and interval[1] <= config['equivalence_margin']})
        elif cell['kind'] == 'truncate' and (root / 'complete.json').exists():
            for split in ('gsm8k', 'symbolic'):
                files = [root.parent / kind / f'step{config["updates"]}' / f'raw_{split}.jsonl'
                         for kind in ('truncate', 'restart', 'random')]
                if not all(path.exists() for path in files):
                    continue
                ordinary = paired_interval(files[0], files[1], config['bootstrap_draws'], config['analysis_seed'])
                random = paired_interval(files[0], files[2], config['bootstrap_draws'], config['analysis_seed'])
                comparisons.append({'cell': cell['slug'], 'split': split, 'kind': 'rank_intervention',
                    'truncate_minus_restart_ci95': ordinary, 'truncate_minus_random_ci95': random,
                    'noninferior': ordinary[0] >= -config['equivalence_margin'],
                    'informed_superior': random[0] > 0})
    field = []
    for model in config.get('field_models', []):
        for seed in config['seeds']:
            root = out / 'field' / model / f'seed{seed}'
            if not (root / 'complete.json').exists():
                missing.append(str(root.relative_to(out)))
                continue
            ordinary = read_json(root / 'ordinary/result.json')['agreement'][config['field_target']]
            for path in sorted(root.glob('*/result.json')):
                result = read_json(path)
                field.append({'model': model, 'seed': seed, 'arm': path.parent.name,
                    'target_gain_over_ordinary': result['agreement'][config['field_target']] - ordinary,
                    'agreement': result['agreement'], 'exposures': result['exposures']})
    result = {'status': 'complete' if not missing else 'partial', 'missing': missing, 'rows': rows,
              'comparisons': comparisons, 'field': field, 'candidate_results': candidate_results,
              'claim_boundary': 'Measured files in this codec grid; selection on calibration only. '
                                'A missing or unreachable cell is not a scientific pass.'}
    write_json(out / 'summary.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/behavioral_trajectory.yaml'))
    parser.add_argument('--out', type=Path, default=Path('reports/behavioral_trajectory_v1'))
    parser.add_argument('--runs', type=Path, default=Path('runs'))
    parser.add_argument('--prepared', type=Path, default=Path('prepared'))
    parser.add_argument('--phase', choices=['prepare', 'check', 'train', 'evaluate', 'report', 'smoke', 'field'], required=True)
    parser.add_argument('--cell', type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.phase == 'prepare':
        lock = prepare(config, args.out, args.runs, args.prepared)
        print(json.dumps({'cells': len(lock['cells']), 'lock': str(args.out / 'lock.json')}))
        return
    lock = read_json(args.out / 'lock.json')
    if lock is None or lock['config'] != config:
        raise ValueError('prepare this exact config before running')
    for path, digest in {**lock['implementation'], **lock['inputs']}.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f'locked input changed: {path}')
    if args.phase == 'check':
        artifacts = {}
        for cell in lock['cells']:
            run = RunSpec.from_dict(cell['run'])
            adapter = args.runs / run.run_id / 'raw_channel.pt'
            if not adapter.is_file():
                raise FileNotFoundError(run.run_id)
            if not Path(run.model.name).is_dir():
                raise FileNotFoundError(run.model.name)
            for path in (adapter, Path(run.model.name) / 'config.json'):
                if str(path) not in artifacts:
                    artifacts[str(path)] = sha256(path)
        previous = read_json(args.out / 'source_artifacts.json')
        if previous is not None and previous != artifacts:
            raise ValueError('source adapters or model configs changed')
        write_json(args.out / 'source_artifacts.json', artifacts)
        print('All source adapters, data hashes and model directories are present.')
    elif args.phase == 'report':
        result = report(lock, args.out)
        print(json.dumps({'status': result['status'], 'missing_cells': len(result['missing'])}))
    elif args.phase == 'field':
        if args.cell is None:
            raise ValueError('choose a field cell')
        matched_field(config, args.out, args.cell)
    else:
        if args.cell is None or not 0 <= args.cell < len(lock['cells']):
            raise ValueError('choose a valid --cell from lock.json')
        cell = lock['cells'][args.cell]
        artifacts = read_json(args.out / 'source_artifacts.json')
        if artifacts is None:
            raise ValueError('run --phase check on the GPU host before execution')
        source = args.runs / cell['run']['run_id'] / 'raw_channel.pt'
        if sha256(source) != artifacts[str(source)]:
            raise ValueError('source adapter changed after preflight')
        if args.phase == 'smoke':
            smoke = {**config, 'updates': 2, 'checkpoints': [1, 2],
                'calibration_rows': 4, 'gsm8k_test_rows': 4, 'ranks': [1, 16],
                'bits': [1], 'max_new_tokens': 32, 'minimum_reference_change': 0,
                'generation_batch_size': 4, 'smoke': True}
            if config.get('scale_controls'):
                smoke['scale_sweep'] = [0.1]
            cell = {**cell, 'kind': 'trajectory', 'slug': cell['slug'] + '/smoke'}
            run_cell(cell, smoke, args.out, args.runs, args.prepared, 'train')
            run_cell(cell, smoke, args.out, args.runs, args.prepared, 'evaluate')
        else:
            run_cell(cell, config, args.out, args.runs, args.prepared, args.phase)


if __name__ == '__main__':
    main()
