import numpy as np
import pytest

from fineqcomp.data import Example
from fineqcomp.information_budget_search import measured_budget, partition
from fineqcomp.information_budget_prediction import fit, predict, curve_forecast, develop, candidates


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
    return dict(run_id=f'{task}_{family}', measures={m: x for m in ('context','online_excess','transfer','shared_gain','gain_integral')},
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
    assert len(candidates()) == 36
    for record in result['nested_predictions']:
        assert record['heldout'] in record['run_id']


def test_artifact_reader_keeps_verification_failures_and_never_opens_later_panels(tmp_path):
    import json
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
    for name, value in [('features.json',f),('targets.json',dict(budgets=[target])),
                        ('data_contract.json',{}),('complete.json',{'complete':True})]:
        (root / name).write_text(json.dumps(value))
    rows, excluded, _ = load_rows(cfg, tmp_path, 'discovery')
    assert len(rows) == 1 and not excluded  # later files intentionally do not exist
    assert rows[0]['verification_retention'] == .75
    assert rows[0]['measures']['online_excess'] == -10.
    assert rows[0]['measures']['transfer'] == 10000.
    assert rows[0]['measures']['shared_gain'] == 15000.
