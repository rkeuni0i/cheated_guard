"""드리프트 모니터링 — PSI (PRD 10.3).

학습 시점 분포(expected) vs 최근 분포(actual)의 안정성을 본다.
PSI < 0.10 안정 / 0.10~0.25 경미 / ≥ 0.25 유의(재학습 트리거).
"""
from __future__ import annotations

import numpy as np

from ..config import PSI_ALERT, PSI_WARN


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")
    breaks = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breaks[0], breaks[-1] = -np.inf, np.inf
    breaks = np.unique(breaks)
    if len(breaks) < 3:
        return 0.0
    e = np.histogram(expected, breaks)[0] / len(expected)
    a = np.histogram(actual, breaks)[0] / len(actual)
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def classify_psi(value: float) -> str:
    if value != value:            # NaN
        return "UNKNOWN"
    if value < PSI_WARN:
        return "STABLE"
    if value < PSI_ALERT:
        return "MINOR"
    return "SIGNIFICANT"


def feature_drift(expected_rows: list[dict], actual_rows: list[dict],
                  features: list[str]) -> dict:
    """피처별 PSI + 등급."""
    out = {}
    for f in features:
        e = np.array([r.get(f, 0.0) for r in expected_rows], dtype=float)
        a = np.array([r.get(f, 0.0) for r in actual_rows], dtype=float)
        val = psi(e, a)
        out[f] = {"psi": round(val, 4), "status": classify_psi(val)}
    return out
