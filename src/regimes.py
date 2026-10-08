"""Stress/calm regime labelling that uses only information available at the time.

The full-sample 90% quantile of the stress signal looks ahead: the threshold at
date t depends on volatility observed after t. Here the threshold at t is the
expanding quantile of the signal over dates strictly before t (phase 0, A5).
"""
from __future__ import annotations

import pandas as pd

MIN_HISTORY = 504  # two years of daily data before the first label


def past_only_stress(signal: pd.Series, quantile: float = 0.90,
                     min_history: int = MIN_HISTORY) -> tuple[pd.Series, pd.Series]:
    """Return ``(stress, valid)``.

    ``stress[t]`` is True when ``signal[t]`` is at or above the ``quantile`` of
    ``signal`` over dates before ``t``. ``valid[t]`` is False during the burn-in
    (fewer than ``min_history`` past observations) and where the signal is NaN;
    callers must restrict regime statistics to ``valid`` dates.
    """
    thr = signal.expanding(min_periods=min_history).quantile(quantile).shift(1)
    valid = thr.notna() & signal.notna()
    stress = (signal >= thr) & valid
    return stress, valid
