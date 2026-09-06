import numpy as np
import pytest

from fineqcomp.predictive_scaling import predict


def test_log_data_extrapolates_known_relation():
    n = np.array([2000, 4000, 8000])
    y = 0.5 + 0.1 * np.log2(n / 2000)
    np.testing.assert_allclose(predict(n, y, [16000, 32000], 'log_data'), [.8, .9])


def test_power_extrapolates_known_relation():
    n = np.array([2000, 4000, 8000])
    np.testing.assert_allclose(predict(n, (n / 2000) ** .2, [32000], 'power'), [16 ** .2])


def test_power_rejects_nonpositive_targets():
    with pytest.raises(ValueError):
        predict([2000, 4000], [0, 1], [8000], 'power')
