import numpy as np
import pandas as pd
import pytest

from src.blocks import block_decomposition, rolling_block_coupling
from src.coupling import rolling_coupling


def _corr(n, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((3 * n, n))
    C = A.T @ A
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def test_parts_add_up_to_total():
    R = _corr(7)
    blocks = {'a': [0, 1, 2], 'b': [3, 4], 'c': [5, 6]}
    d = block_decomposition(R, blocks)
    intra = d['intra_a'] + d['intra_b'] + d['intra_c']
    assert d['total'] == pytest.approx(intra + d['inter'], abs=1e-12)
    assert d['inter'] >= 0
    for k in ('mi_a__b', 'mi_a__c', 'mi_b__c'):
        assert d[k] >= -1e-12


def test_independent_blocks_have_no_between_coupling():
    R = np.eye(6)
    R[np.ix_([0, 1, 2], [0, 1, 2])] = _corr(3, seed=1)
    R[np.ix_([3, 4, 5], [3, 4, 5])] = _corr(3, seed=2)
    d = block_decomposition(R, {'a': [0, 1, 2], 'b': [3, 4, 5]})
    assert d['inter'] == pytest.approx(0.0, abs=1e-12)
    assert d['mi_a__b'] == pytest.approx(0.0, abs=1e-12)


def test_two_blocks_inter_equals_pairwise_mi():
    R = _corr(5, seed=3)
    d = block_decomposition(R, {'a': [0, 1], 'b': [2, 3, 4]})
    assert d['inter'] == pytest.approx(d['mi_a__b'], abs=1e-12)


def test_blocks_must_partition_columns():
    with pytest.raises(ValueError):
        block_decomposition(np.eye(4), {'a': [0, 1], 'b': [1, 2, 3]})


def test_rolling_total_matches_rank_h_dep():
    rng = np.random.default_rng(4)
    idx = pd.bdate_range('2001-01-01', periods=300)
    lr = pd.DataFrame(rng.standard_normal((300, 6)) * 0.01, index=idx,
                      columns=list('abcdef'))
    dec = rolling_block_coupling(lr, {'x': ['a', 'b', 'c'], 'y': ['d', 'e', 'f']}, 252)
    ref = rolling_coupling(lr, 252)
    np.testing.assert_allclose(dec['total'], -ref['h_dep_rank'], rtol=1e-10, atol=1e-12)
