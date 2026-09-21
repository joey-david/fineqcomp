from pathlib import Path
import yaml
from fineqcomp.rule_discovery import cue_rows


def test_wrong_cue_changes_input_but_never_oracle_label():
    source = Path(__file__).resolve().parents[3]
    info = yaml.safe_load((source / 'configs/design_2026_09_07/f10_storage_versus_discovery.yaml').read_text())
    info['codebook_dir'] = str(source / 'codebooks')
    hidden, correct, wrong = [cue_rows(info, 4, 11, mode) for mode in ('hidden', 'correct', 'wrong')]
    assert [r.response for r in hidden] == [r.response for r in correct] == [r.response for r in wrong]
    assert all('T?' in r.prompt for r in hidden)
    assert sum(r.metadata['cue_disagrees'] for r in wrong) > 600
    assert all(w.metadata['cued_label'] == correct[((w.metadata['family'] + 1) % 4) * 16 + w.metadata['item']].metadata['label_index'] for w in wrong)
