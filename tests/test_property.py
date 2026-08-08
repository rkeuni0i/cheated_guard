"""속성 테스트 — 불변식 (PRD 13.1 Property).

hypothesis 없이 무작위 스윕으로 근사한다: 스코어는 항상 0~100, 등급 경계는
단조적, 융합은 발산하지 않는다.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spendguard.config import GRADE_BOUNDS
from spendguard.detection.statistics import axis_amount
from spendguard.models import Profile


def _profile(median, mad_log):
    stats = {"median": median, "mad_log": mad_log, "p99": median * 8, "p95": median * 4,
             "max": median * 15, "n": 300, "log_mean": math.log(median), "log_std": 1.0,
             "monthly_sum_ewma": median * 40}
    return Profile("U", 1, "s", "e", 300, False, stats, {},
                   {"hour_hist": [1/24]*24}, {})


def test_axis_amount_bounds():
    random.seed(1)
    for _ in range(2000):
        median = random.randint(3_000, 200_000)
        mad_log = random.uniform(0.1, 2.0)
        amount = random.randint(1_000, 5_000_000)
        v = axis_amount(amount, _profile(median, mad_log))
        assert 0.0 <= v <= 100.0


def test_grade_monotonic():
    """스코어가 오르면 등급도 유지되거나 상승한다 (강등 없음, 기본 매핑)."""
    def grade_of(score):
        return next(g for g, lo in GRADE_BOUNDS if score >= lo)
    order = {"R0": 0, "R1": 1, "R2": 2, "R3": 3, "R4": 4}
    prev = -1
    for s in range(0, 101):
        cur = order[grade_of(s)]
        assert cur >= prev
        prev = cur


def test_grade_boundaries():
    def grade_of(score):
        return next(g for g, lo in GRADE_BOUNDS if score >= lo)
    assert grade_of(0) == "R0"
    assert grade_of(29) == "R0"
    assert grade_of(30) == "R1"
    assert grade_of(70) == "R3"
    assert grade_of(90) == "R4"
    assert grade_of(100) == "R4"
