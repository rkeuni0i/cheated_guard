"""단위 테스트 — 피처·통계·룰·정규화 (PRD 13.1 Unit)."""
from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spendguard.detection.rules import score_l1
from spendguard.detection.statistics import axis_amount, _empirical_percentile
from spendguard.ingestion.normalizer import normalize_merchant
from spendguard.ingestion.validator import validate
from spendguard.models import Profile, Transaction
from spendguard.profiling.special_merchants import classify_variance, is_routine
from spendguard.taxonomy import mcc_to_category

KST = timezone(timedelta(hours=9))


def _profile(median=20_000, mad_log=0.9, p99=145_000):
    stats = {"median": median, "mad_log": mad_log, "p99": p99, "p95": 68_000,
             "max": 289_000, "n": 400, "log_mean": math.log(median), "log_std": 1.0,
             "monthly_sum_ewma": 640_000}
    return Profile("U-1", 1, "2026-05-01", "2026-08-01", 400, False,
                   stats, {}, {"hour_hist": [1/24]*24, "dow_hist": [1/7]*7,
                               "night_ratio": 0.03}, {})


# ── 정규화 (DQ / 4.6) ──────────────────────────────────────────────────
def test_normalize_merchant():
    assert normalize_merchant("GS25 강남역점") == "gs25"
    assert normalize_merchant("(주)올리브영 홍대점") == "올리브영"
    assert normalize_merchant("") == "UNKNOWN_MERCHANT"


def test_mcc_mapping():
    assert mcc_to_category("5732") == "ELECTRONICS"
    assert mcc_to_category("5814") == "CAFE"
    assert mcc_to_category("9999") == "ETC"
    assert mcc_to_category("3500") == "TRAVEL"     # 3000-3999 범위


# ── DQ 검증 ────────────────────────────────────────────────────────────
def test_validate_amount():
    tx = Transaction("U-1", "C-1", "A1", -100, datetime.now(KST), "m", "5812")
    assert validate(tx).code == "DQ-01"


def test_validate_future():
    tx = Transaction("U-1", "C-1", "A1", 1000,
                     datetime.now(KST) + timedelta(days=2), "m", "5812")
    assert validate(tx).code == "DQ-02"


# ── 통계 축 (6.3) ──────────────────────────────────────────────────────
def test_axis_amount_monotonic():
    p = _profile()
    lo = axis_amount(20_000, p)
    hi = axis_amount(300_000, p)
    assert 0 <= lo <= hi <= 100
    assert lo < 5                        # median 근처는 정상
    assert hi > 30                       # 15배는 뚜렷한 이탈 (분산 큰 사용자)


def test_empirical_percentile():
    ref = {"median": 20_000, "p95": 68_000}
    assert _empirical_percentile(20_000, ref) < 0.6
    assert _empirical_percentile(300_000, ref) > 0.95


# ── L1 수확 체감 (6.2) ─────────────────────────────────────────────────
def test_score_l1_diminishing():
    r1 = [{"id": "R-01", "weight": 28, "severity": "HIGH", "label": "x"}]
    r2 = r1 + [{"id": "R-02", "weight": 25, "severity": "HIGH", "label": "y"}]
    r3 = r2 + [{"id": "R-05", "weight": 20, "severity": "HIGH", "label": "z"}]
    s1, _ = score_l1(r1)
    s2, _ = score_l1(r2)
    s3, _ = score_l1(r3)
    assert 0 <= s1 <= s2 <= s3 <= 100
    # 수확 체감: 세 번째 룰의 한계 기여가 두 번째보다 작다
    assert (s3 - s2) < (s2 - s1)


def test_score_l1_empty():
    assert score_l1([]) == (0, [])


# ── 특수 가맹점 분류 (4.5) ─────────────────────────────────────────────
def test_classify_variance():
    # 균일 → LOW
    assert classify_variance([1500] * 50) == "LOW"
    # 고분산(올리브영형: 소액 다수 + 가끔 고액) → HIGH
    import numpy as np
    rng = np.random.default_rng(0)
    amts = list(np.round(rng.lognormal(np.log(30_000), 1.5, 200)).astype(int))
    assert classify_variance(amts) == "HIGH"
    # 표본 부족 → MID
    assert classify_variance([1000, 2000]) == "MID"


def test_is_routine():
    days = [30, 30, 31, 29]
    amounts = [39000, 39500, 39200, 39100]
    assert is_routine(days, amounts) is True
    assert is_routine([1, 40, 3], [1000, 500000, 200]) is False
