from pathlib import Path

import numpy as np
import pytest
import yaml

from fineqcomp.config import ModelSpec
from fineqcomp.data import Example
from fineqcomp.information_budget_search import (
    budget_interval, exposure_record, measured_budget, partition)
from fineqcomp.information_budget_prediction import (
    MEASURES, fit, predict, curve_forecast, develop, candidates, exposure_summary, screening_report)


def test_dialogue_rate_panel_scores_held_out_dialogue():
    raw = yaml.safe_load(
        Path("configs/information_budget_search/dialogue_rate_data.yaml").read_text()
    )
    dialogue = raw["datasets"]["kind_dialogue"]

    assert dialogue["test_rows"] >= 128
    assert dialogue["evaluations"] == [
        {
            "key": "kind_dialogue",
            "path": "Anthropic/hh-rlhf",
            "name": "default",
            "revision": "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa",
            "split": "test",
            "converter": "hh_rlhf",
        }
    ]


def test_source_augmentations_never_cross_halves():
    rows = [Example(str(i), f'variant {i}', 'answer', {'source_problem': str(i // 3)}) for i in range(90)]
    left, right = partition(rows, 81)
    assert len(left) + len(right) == len(rows)
    assert {r.metadata['source_problem'] for r in left}.isdisjoint(r.metadata['source_problem'] for r in right)
    assert partition(list(reversed(rows)), 81) == [left, right]


def test_budget_uses_achieved_file_and_preserves_failed_verification():
    base = dict(selection=100., verification=100.)
    raw = dict(selection=0., verification=0.)
    grid = [dict(key='cheap', scale=1., file_bits=808, selection=30., verification=40.),
            dict(key='large', scale=1., file_bits=1608, selection=8., verification=25.),
            dict(key='scaled', scale=.5, file_bits=608, selection=1., verification=1.)]
    target = measured_budget(grid, base, raw, .9)
    assert target['file_bits'] == 1608  # not an interpolated crossing
    assert target['verification_retention'] == .75
    assert not target['below_smallest_tested']
    assert measured_budget(grid, base, raw, .9, scaled=True)['file_bits'] == 608
    assert measured_budget(grid, base, raw, .99)['status'] == 'above_grid'
    assert measured_budget(grid, base, base, .9)['status'] == 'no_positive_reference_gain'


def row(x, y, task='a', family='m'):
    return dict(run_id=f'{task}_{family}', measures={m: x for m in MEASURES},
        container_bits=128., target_bits=y, task=task, family=family,
        forecasts={f: y for f in ('early_identity','inverse_sqrt','inverse_time')})


def test_calibration_extrapolates_without_receiver_intercept_or_test_target():
    train = [row(x, 7*x**.5) for x in (4., 16., 64.)]
    fitted = fit(train, dict(measure='transfer', law='power', container=False))
    test = row(256., 1234567.)
    assert predict(fitted, test) == pytest.approx(112.)
    test['target_bits'] = -100
    assert predict(fitted, test) == pytest.approx(112.)
    assert fit([row(-1., 7.)], dict(measure='context', law='power', container=False)) is None


def test_curve_forecast_uses_early_losses_and_measured_bytes():
    traces = []
    for replica in (0, 1):
        for t in (8, 32, 128):
            traces.append(dict(replica=replica, step=t, reference={'selection': 10 + 64/t}, grid=[
                dict(key='small', file_bits=801, selection=40 + 64/t),
                dict(key='large', file_bits=1601, selection=12 + 64/t)]))
    features = dict(traces=traces, base_rate={'selection':100}, full_horizon=1024)
    assert curve_forecast(features, 'inverse_time') == 1601
    assert curve_forecast(features, 'early_identity') == 1601


def test_nested_selection_keeps_whole_tasks_and_families_out():
    rows = [row(float(i+1), float((i+1)*800), task=str(i), family=m) for i in range(4) for m in ('m', 'q')]
    result = develop(rows)
    assert len(result['nested_predictions']) == 16
    assert result['nested_metrics']['covered'] == 16
    assert len(candidates()) == 42
    for record in result['nested_predictions']:
        assert record['heldout'] in record['run_id']


def test_artifact_reader_keeps_verification_failures_and_never_opens_later_panels(tmp_path):
    import json
    import torch
    from fineqcomp.codec import encode_tensor_map
    from fineqcomp.information_budget_prediction import load_rows
    cfg = {'cells': [dict(stage=stage, run=dict(run_id=stage, model={'key':'mistral_7b_base'}, dataset_key='code'))
                     for stage in ('discovery','development','outer_audit')]}
    root = tmp_path / 'cells/discovery'; root.mkdir(parents=True)
    feature = dict(bits=270., examples=2, per_example_bits=[90.,180.])
    grid = [dict(key='r1_b16_s1', rank=1, precision=16, scale=1., file_bits=1608, selection=460.)]
    f = dict(base=dict(bits=300., per_example_bits=[100.,200.]), base_rate={'selection':500.},
             base_halves=[dict(bits=500.),dict(bits=500.)], training_rows=1000, full_horizon=2000,
             frozen_context=[dict(requested=4,gain_bits=10.)], online_excess_bits=-10.,
             traces=[dict(replica=r,step=t,feature=feature,other_half=dict(bits=480.,examples=2),
                          reference={'selection':450.},grid=grid) for r in (0,1) for t in (8,32,128)])
    target = dict(retention=.9,scaled=False,status='measured',file_bits=3208,
                  verification_retention=.75,below_smallest_tested=False)
    storage = encode_tensor_map({'layer.lora_A.default.weight': torch.zeros(1, 32)},
        root / 'replica0/step8/codecs/r1_b16_s1.fqcb', 16)
    grid[0]['file_bits'] = storage['file_bits']
    for name, value in [('features.json',f),('targets.json',dict(budgets=[target])),
                        ('data_contract.json',{}),('complete.json',{'complete':True})]:
        (root / name).write_text(json.dumps(value))
    rows, excluded, _ = load_rows(cfg, tmp_path, 'discovery')
    assert len(rows) == 1 and not excluded  # later files intentionally do not exist
    assert rows[0]['verification_retention'] == .75
    assert rows[0]['measures']['online_excess'] == -10.
    assert rows[0]['measures']['transfer'] == 10000.
    assert rows[0]['measures']['shared_gain'] == 15000.


def test_public_capacity_is_independent_of_weight_entropy(tmp_path):
    import torch
    from fineqcomp.codec import encode_tensor_map
    from fineqcomp.information_budget_prediction import public_container_bits
    torch.manual_seed(7)
    sizes, capacities = [], []
    for name, tensor in [('zeros', torch.zeros(1, 512)), ('random', torch.randn(1, 512))]:
        path = tmp_path / f'{name}.fqcb'
        record = encode_tensor_map({'layer.lora_A.default.weight': tensor}, path, 16)
        sizes.append(record['file_bits'])
        capacities.append(public_container_bits(path))
    assert sizes[0] != sizes[1]
    assert capacities[0] == capacities[1]


class CharTokenizer:
    eos_token_id = 1
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return ([2] if add_special_tokens else []) + [10 + (ord(c) % 50) for c in text]


def exposure_for(train_limit, score_limit, byte_filter=8):
    from fineqcomp.modeling import CausalExampleDataset
    model = ModelSpec('m', 'model', 'revision', 'bf16')
    rows = [Example('short', 'abc', 'defg', {}),           # 4 prompt + 5 response tokens
            Example('cut', 'abcdefgh', 'ij', {}),          # 9 prompt + 3 response tokens
            Example('prompt_fills', 'abcdefghijkl', 'm', {})]  # 13 prompt + 2 response tokens
    data = CausalExampleDataset(CharTokenizer(), rows, model, train_limit)
    return exposure_record(data, rows, train_limit, score_limit, byte_filter)


def test_exposure_separates_dropped_rows_from_a_cut_response():
    record = exposure_for(10, 32)
    assert record['original_rows'] == 3 and record['usable_rows'] == 2
    assert record['dropped_by_length'] == 1  # the prompt alone reached the limit
    assert record['dropped_by_span'] == 0 and record['dropped_without_marker'] == 0
    assert record['rows_truncated'] == 1  # exact, not a row sitting at the limit
    assert record['supervised_tokens'] == 6
    assert record['response_tokens'] == 8
    assert record['unsupervised_response_tokens'] == 2
    assert record['rows_over_byte_filter'] == 2


def test_scored_span_exceeds_the_supervised_span_at_a_longer_limit():
    wide = exposure_for(10, 32)
    assert wide['scored_tokens'] == wide['response_tokens']
    assert wide['scored_tokens'] > wide['supervised_tokens']
    assert wide['rows_over_score_limit'] == 0
    narrow = exposure_for(10, 11)
    assert narrow['scored_tokens'] == 7  # the scorer cuts what training already cut
    assert narrow['rows_over_score_limit'] == 1
    matched = exposure_for(32, 32)
    assert matched['supervised_tokens'] == matched['scored_tokens']
    assert matched['unsupervised_response_tokens'] == 0


def test_exposure_summary_keeps_cells_that_never_reached_a_budget():
    measured = dict(run_id='learned', exposure=exposure_for(10, 32))
    failed = dict(run_id='no_budget', exposure=exposure_for(10, 11))
    unknown = dict(run_id='older_artifact', exposure=None)
    summary = exposure_summary([measured, failed, unknown])
    assert [c['run_id'] for c in summary['cells']] == ['learned', 'no_budget']
    assert summary['missing'] == 1
    assert summary['limits_matched'] is False  # trained at 10, scored at 32
    assert summary['worst_unsupervised_fraction'] == 0.25
    assert summary['cells'][0]['dropped_rows'] == 1
    assert summary['cells'][0]['truncated_fraction'] == 0.5
    assert exposure_summary([])['limits_matched'] is None


def screening_panel(signal, seed=1):
    """Two families by six tasks, the shape of the discovery panel."""
    generator = np.random.default_rng(seed)
    rows = []
    for family in ('mistral', 'qwen'):
        for task in ('code', 'dialogue', 'summary', 'math', 'sql', 'xbrl'):
            x = float(generator.uniform(4, 400))
            y = 7000 * x ** .5 if signal else float(generator.uniform(1e4, 4e5))
            rows.append(dict(run_id=f'{family}_{task}', task=task, family=family,
                measures={m: x for m in MEASURES},
                container_bits=128000., target_bits=y,
                forecasts={f: y if signal else float(generator.uniform(1e4, 4e5))
                           for f in ('early_identity', 'inverse_sqrt', 'inverse_time')}))
    return rows


def test_screening_exposes_the_optimism_of_searching_many_rules_on_twelve_cells():
    report = screening_report(screening_panel(signal=False), repeats=60, seed=3)
    assert report['candidates'] == 42 and report['cells'] == 12
    # The search minimum flatters itself; re-running selection inside every fold does not.
    assert report['nested']['mean_absolute_log2_error'] > report['optimistic_mean_absolute_log2_error']
    # Against no signal the winner does not beat predicting the mean, and the
    # permutation null reaches the same error without any relationship to fit.
    assert report['margin_over_best_baseline'] <= 0
    assert report['null']['p_value'] > .05
    assert report['stability']['distinct_winners'] > 1


def test_screening_separates_a_real_relationship_from_selection_advantage():
    report = screening_report(screening_panel(signal=True), repeats=60, seed=3)
    assert report['null']['p_value'] < .05
    assert report['null']['observed'] < report['null']['null_best']
    assert report['margin_over_best_baseline'] > 0
    assert report['stability']['distinct_winners'] == 1
    assert report['stability']['modal_winner'] == report['stability']['full_sample_winner']
    assert report['stage'] == 'screening'


def codec_grid(examples=48, seed=0):
    """A base, a reference and a rate grid, all as per-example bits."""
    generator = np.random.default_rng(seed)
    base = {split: dict(per_example_bits=list(generator.uniform(9., 11., examples) + 4.))
            for split in ('selection', 'verification')}
    reference = {}
    for split in ('selection', 'verification'):
        reference[f'{split}_per_example'] = [b - 4. for b in base[split]['per_example_bits']]
    grid = []
    for index, (bits, penalty) in enumerate([(800, 3.6), (1600, 2.0), (3200, .8), (6400, .1)]):
        row = dict(key=f'c{index}', rank=1, precision=16, scale=1., file_bits=bits)
        for split in ('selection', 'verification'):
            row[f'{split}_per_example'] = [b - 4. + penalty for b in base[split]['per_example_bits']]
        row.update({s: sum(row[f'{s}_per_example']) for s in ('selection', 'verification')})
        grid.append(row)
    return grid, base, reference


def test_budget_interval_reports_the_codec_spread_behind_one_achieved_file():
    grid, base, reference = codec_grid()
    point = measured_budget(grid, {s: sum(base[s]['per_example_bits']) for s in ('selection', 'verification')},
                            {s: sum(reference[f'{s}_per_example']) for s in ('selection', 'verification')}, .9)
    assert point['status'] == 'measured' and point['file_bits'] == 6400
    spread = budget_interval(grid, base, reference, .9, resamples=120, seed=1)
    assert spread['resamples'] == 120
    assert spread['measured_fraction'] == 1.
    assert spread['file_bits_p05'] <= spread['file_bits_median'] <= spread['file_bits_p95']
    assert spread['file_bits_p95'] >= point['file_bits']
    assert spread['verification_reported'] == 120


def test_budget_interval_keeps_undefined_targets_and_verification_failures():
    grid, base, reference = codec_grid()
    # A reference that never beats the base leaves the 90% target undefined, and
    # resampling must report that rather than returning a number anyway.
    flat = {f'{s}_per_example': list(base[s]['per_example_bits']) for s in ('selection', 'verification')}
    absent = budget_interval(grid, base, flat, .9, resamples=60, seed=2)
    assert absent['measured_fraction'] == 0.
    assert absent['status_counts'] == {'no_positive_reference_gain': 60}
    assert absent['file_bits_median'] is None
    # A grid that cannot reach the target is censoring, not a missing measurement.
    steep = budget_interval(grid, base, reference, .999, resamples=60, seed=2)
    assert steep['status_counts'].get('above_grid', 0) > 0
    assert budget_interval([], base, reference, .9)['status'] == 'no_codecs'


def test_redundant_measures_are_reported_as_one_scalar_not_five_theories():
    from fineqcomp.information_budget_prediction import measure_redundancy
    rows = screening_panel(signal=True)  # every measure carries the same value
    report = measure_redundancy(rows)
    assert all(abs(v) >= .95 for v in report['pairwise_spearman'].values())
    assert len(report['indistinguishable_pairs']) == len(report['pairwise_spearman']) == 15
    assert report['spread_of_nested_error'] == 0.  # no measure predicts what the others cannot


def test_a_measure_that_carries_its_own_signal_separates_from_the_others():
    rows = screening_panel(signal=True)
    generator = np.random.default_rng(5)
    for row in rows:  # break every measure except `transfer`
        for name in ('context', 'online_excess', 'shared_gain', 'gain_integral', 'spectral'):
            row['measures'][name] = float(generator.uniform(1., 400.))
        row['forecasts'] = {f: float(generator.uniform(1e4, 4e5)) for f in row['forecasts']}
    from fineqcomp.information_budget_prediction import measure_redundancy
    report = measure_redundancy(rows)
    assert not report['indistinguishable_pairs']
    assert report['spread_of_nested_error'] > 0
    best = min(report['nested_alone'], key=lambda m: report['nested_alone'][m]['mean_absolute_log2_error'])
    assert best == 'transfer'


def test_spectral_bits_measure_a_size_not_a_likelihood_change():
    import torch
    from fineqcomp.codec import pad_lora_rank, spectral_bits, truncate_lora_rank
    torch.manual_seed(3)
    width, height, rank = 64, 96, 16
    def update(true_rank):
        a, b = torch.zeros(rank, width), torch.zeros(height, rank)
        a[:true_rank] = torch.randn(true_rank, width)
        b[:, :true_rank] = torch.randn(height, true_rank)
        return {'m.lora_A.default.weight': a, 'm.lora_B.default.weight': b}
    sizes = [spectral_bits(update(r)) for r in (1, 2, 4, 8, 16)]
    assert sizes == sorted(sizes)
    assert sizes[0] == pytest.approx(16 * (width + height), rel=1e-6)  # one direction, rank one
    # The container is not the code: padding a truncated update back to its
    # trained width leaves the measure alone, which a file size would not.
    narrow = truncate_lora_rank(update(16), 1)
    assert spectral_bits(pad_lora_rank(narrow, rank)) == pytest.approx(spectral_bits(narrow))
    assert spectral_bits({'m.lora_A.default.weight': torch.zeros(rank, width),
                          'm.lora_B.default.weight': torch.zeros(height, rank)}) == 0.


def test_a_measure_absent_from_older_artifacts_is_ineligible_rather_than_wrong():
    rows = screening_panel(signal=True)
    for row in rows:
        row['measures']['spectral'] = None
    fitted = fit(rows, dict(measure='spectral', law='power', container=False))
    assert fitted is None  # no fit, so no candidate, rather than a silent zero
    assert predict(None, rows[0]) is None


def test_zero_point_separates_a_base_that_knew_it_from_a_fine_tune_that_failed():
    from fineqcomp.information_budget_prediction import zero_point
    measured = dict(run_id='learned', task='code', base_bits=500., reference_bits=400.)
    # Same task, no gain, but a peer gained: this base had less to learn.
    idle = dict(run_id='idle', task='code', base_bits=500., reference_bits=500.,
                reason='no_positive_reference_gain')
    # Different task where nothing gained at all: the task, not the receiver.
    dead_a = dict(run_id='dead_a', task='xbrl', base_bits=500., reference_bits=500.,
                  reason='no_positive_reference_gain')
    dead_b = dict(run_id='dead_b', task='xbrl', base_bits=500., reference_bits=520.,
                  reason='no_positive_reference_gain')
    coarse = dict(run_id='coarse', task='code', base_bits=500., reference_bits=300.,
                  reason='above_grid')
    report = zero_point([measured], [idle, dead_a, dead_b, coarse])
    assert report['verdicts']['learned'] == 'measured'
    assert report['verdicts']['idle'] == 'no_gain_while_peers_gained'
    assert report['verdicts']['dead_a'] == 'no_gain_and_no_peer_gained'
    assert report['verdicts']['dead_b'] == 'reference_worse_than_base'
    assert report['verdicts']['coarse'] == 'gained_but_above_grid'
    assert report['counts']['no_gain_while_peers_gained'] == 1


def test_zero_point_says_unknown_rather_than_guessing_without_an_anchor():
    from fineqcomp.information_budget_prediction import zero_point
    report = zero_point([], [dict(run_id='older', task='code', reason='no_positive_reference_gain')])
    assert report['verdicts']['older'] == 'unknown_no_anchor'
