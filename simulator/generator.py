"""합성 거래 데이터 생성기 (PRD 부록 A).

생성 가정을 코드로 명시하는 것이 이 모듈의 목적이다(부록 A 취지).
- 금액: 카테고리별 로그정규 분포
- 시간: 점심/저녁 이중 봉우리 + 균등 꼬리
- 가맹점: Zipf 분포(단골 소수 + 롱테일)
- 정기 거래: 통신/구독/교통 주입
- 계절성: 12월 1.25배, 5월 1.15배, 2월 0.9배
- 이상 시나리오: SC-01~07(라벨 True) 0.8%, SC-08~12(hard negative) 2%

성능 주장 시 반드시 "합성 데이터 기준"임을 밝혀야 한다(부록 A 경고).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from spendguard.models import Transaction
from spendguard.taxonomy import CATEGORIES

KST = timezone(timedelta(hours=9))

# ── 아키타입 (PRD 부록 A) ──────────────────────────────────────────────
USER_ARCHETYPES = {
    "student":    dict(monthly=(300_000, 800_000),   tpm=(40, 80),
                       online=0.45, variance=1.0),
    "office":     dict(monthly=(800_000, 2_000_000), tpm=(50, 100),
                       online=0.35, variance=1.0),
    "freelancer": dict(monthly=(600_000, 3_000_000), tpm=(30, 70),
                       online=0.55, variance=1.35),
    "senior":     dict(monthly=(500_000, 1_500_000), tpm=(25, 50),
                       online=0.12, variance=1.0),
}

# 카테고리별 (중앙금액, MCC, 채널선호) — 로그정규 mu=ln(median)
_CAT_PROFILE = {
    "FOOD":        (12_000, "5812"), "CAFE": (5_500, "5814"),
    "GROCERY":     (38_000, "5411"), "TRANSPORT": (1_500, "4111"),
    "FUEL":        (60_000, "5541"), "BEAUTY": (34_000, "5977"),
    "FASHION":     (68_000, "5651"), "ELECTRONICS": (42_000, "5732"),
    "ECOMMERCE":   (28_000, "5964"), "MEDICAL": (18_000, "8062"),
    "ENTERTAIN":   (22_000, "7832"), "SUBSCRIPTION": (11_000, "5968"),
    "TRAVEL":      (120_000, "7011"),
}
_SIGMA = {"LOW": 0.5, "MID": 0.9, "HIGH": 1.4}

# 아키타입별 카테고리 소비 비중 (합=1로 정규화)
_CAT_WEIGHTS = {
    "student":    {"FOOD": .28, "CAFE": .15, "TRANSPORT": .12, "ECOMMERCE": .12,
                   "ENTERTAIN": .1, "FASHION": .08, "SUBSCRIPTION": .06,
                   "GROCERY": .05, "BEAUTY": .04},
    "office":     {"FOOD": .22, "CAFE": .12, "GROCERY": .14, "TRANSPORT": .08,
                   "FUEL": .08, "ECOMMERCE": .1, "FASHION": .08, "BEAUTY": .06,
                   "SUBSCRIPTION": .05, "ENTERTAIN": .04, "MEDICAL": .03},
    "freelancer": {"FOOD": .18, "CAFE": .14, "ECOMMERCE": .16, "ELECTRONICS": .1,
                   "FUEL": .08, "GROCERY": .1, "SUBSCRIPTION": .08, "FASHION": .08,
                   "TRAVEL": .04, "ENTERTAIN": .04},
    "senior":     {"GROCERY": .24, "FOOD": .18, "MEDICAL": .14, "FUEL": .1,
                   "TRANSPORT": .08, "FASHION": .08, "SUBSCRIPTION": .06,
                   "CAFE": .06, "ENTERTAIN": .06},
}

# 카테고리별 가맹점 풀 (Zipf 추출 대상)
_MERCHANTS = {
    "FOOD": ["김밥천국", "본죽", "맘스터치", "롯데리아", "한솥도시락", "명동교자", "노브랜드버거"],
    "CAFE": ["스타벅스", "투썸플레이스", "이디야", "메가커피", "빽다방", "폴바셋"],
    "GROCERY": ["이마트", "홈플러스", "롯데마트", "노브랜드", "GS더프레시", "다이소"],
    "TRANSPORT": ["티머니충전", "서울지하철", "카카오T", "코레일"],
    "FUEL": ["GS칼텍스", "SK에너지", "S-OIL", "현대오일뱅크"],
    "BEAUTY": ["올리브영", "아리따움", "롭스", "이니스프리", "다이소뷰티"],
    "FASHION": ["무신사", "자라", "유니클로", "탑텐", "스파오", "지오다노"],
    "ELECTRONICS": ["컴퓨존", "다나와", "하이마트", "삼성디지털프라자", "애플스토어"],
    "ECOMMERCE": ["쿠팡", "네이버", "11번가", "G마켓", "옥션", "티몬"],
    "MEDICAL": ["연세내과", "365약국", "튼튼정형외과", "온누리약국"],
    "ENTERTAIN": ["CGV", "메가박스", "넷플릭스", "PC방", "노래방"],
    "SUBSCRIPTION": ["SKT통신", "KT통신", "넷플릭스구독", "유튜브프리미엄", "쿠팡와우"],
    "TRAVEL": ["대한항공", "야놀자", "여기어때", "아고다", "제주항공"],
}

_REGIONS = ["서울 강남구", "서울 마포구", "서울 송파구", "경기 성남시", "경기 수원시",
            "부산 해운대구", "인천 남동구", "대구 수성구"]


@dataclass
class GenTx:
    tx: Transaction
    label: Optional[bool] = None      # None=미주입, True=이상, False=hard negative
    scenario: Optional[str] = None
    event_id: Optional[int] = None    # 같은 사기 '사건'에 속한 거래 묶음


def _zipf_pick(rng, items, s=1.1):
    n = len(items)
    weights = 1.0 / np.power(np.arange(1, n + 1), s)
    weights /= weights.sum()
    return items[rng.choice(n, p=weights)]


def _lognormal_amount(rng, category, variance_mult=1.0) -> int:
    median, _ = _CAT_PROFILE[category]
    vc = CATEGORIES[category][1]
    # sigma 상한: 현실적 롱테일 유지하되 수백만원대 이상치 폭주를 막는다
    sigma = min(_SIGMA[vc] * variance_mult, 1.2)
    amt = rng.lognormal(mean=np.log(median), sigma=sigma)
    amt = min(max(1000, round(amt / 10) * 10), 1_500_000)
    return int(amt)


def _sample_hour(rng) -> float:
    r = rng.random()
    if r < 0.45:
        return float(np.clip(rng.normal(12.5, 1.8), 6, 23))
    if r < 0.90:
        return float(np.clip(rng.normal(19, 2.2), 6, 23))
    return float(rng.uniform(7, 23))


def _season_mult(month: int) -> float:
    return {12: 1.25, 5: 1.15, 2: 0.9}.get(month, 1.0)


def _make_tx(rng, user_id, card_id, category, when, online_ratio,
             variance_mult=1.0, amount=None, channel=None, mcc=None,
             merchant=None) -> Transaction:
    _, default_mcc = _CAT_PROFILE[category]
    merchant = merchant or _zipf_pick(rng, _MERCHANTS[category])
    if channel is None:
        channel = "ONLINE" if rng.random() < online_ratio else "OFFLINE"
    if amount is None:
        amount = _lognormal_amount(rng, category, variance_mult)
    amount = int(round(amount * _season_mult(when.month)))
    amount = max(1000, round(amount / 10) * 10)
    return Transaction(
        user_id=user_id, card_id=card_id,
        approval_no=uuid.uuid4().hex[:12],
        amount=amount, approved_at=when,
        merchant_name=merchant, mcc=mcc or default_mcc,
        channel=channel,
        merchant_region=rng.choice(_REGIONS) if channel != "ONLINE" else None,
    )


def generate(n_users: int = 200, months: int = 12, seed: int = 42,
             end_date: Optional[datetime] = None,
             anomaly_rate: float = 0.008,
             hard_negative_rate: float = 0.02) -> tuple[list[dict], list[GenTx]]:
    """사용자 목록과 거래 리스트(라벨 포함)를 생성한다.

    반환: (users, gen_txs)
      users: [{"user_id", "archetype", "sensitivity", "card_id"}...]
      gen_txs: 시간 오름차순 정렬된 GenTx 리스트
    """
    rng = np.random.default_rng(seed)
    end_date = end_date or datetime(2026, 8, 8, tzinfo=KST)
    start_date = end_date - timedelta(days=30 * months)

    users: list[dict] = []
    all_txs: list[GenTx] = []
    archetypes = list(USER_ARCHETYPES.keys())

    for i in range(n_users):
        arche = archetypes[i % len(archetypes)]
        cfg = USER_ARCHETYPES[arche]
        user_id = f"U-{i:04d}"
        card_id = f"C-{i:04d}"
        sensitivity = rng.choice(["LOW", "DEFAULT", "DEFAULT", "HIGH"])
        users.append(dict(user_id=user_id, archetype=arche,
                          sensitivity=sensitivity, card_id=card_id))

        tpm = rng.integers(cfg["tpm"][0], cfg["tpm"][1] + 1)
        cat_w = _CAT_WEIGHTS[arche]
        cats = list(cat_w.keys())
        probs = np.array([cat_w[c] for c in cats])
        probs /= probs.sum()

        user_txs = _gen_user_normal(rng, user_id, card_id, cats, probs,
                                    cfg, tpm, start_date, months)
        _inject_recurring(rng, user_id, card_id, start_date, months, user_txs)
        all_txs.extend(GenTx(tx=t) for t in user_txs)

    # 이상/hard-negative 주입
    travel_windows: list[tuple] = []
    _inject_anomalies(rng, users, all_txs, anomaly_rate, hard_negative_rate,
                      start_date, end_date, travel_windows)

    all_txs.sort(key=lambda g: g.tx.approved_at)
    return users, all_txs, travel_windows


def _gen_user_normal(rng, user_id, card_id, cats, probs, cfg, tpm,
                     start_date, months) -> list[Transaction]:
    txs: list[Transaction] = []
    for m in range(months):
        month_start = start_date + timedelta(days=30 * m)
        month_tx = max(1, int(rng.normal(tpm, tpm * 0.15)))
        for _ in range(month_tx):
            day = int(rng.integers(0, 30))
            hour = _sample_hour(rng)
            # 요일 가중: 금·토 1.3배 (샘플 재추출로 근사)
            when = month_start + timedelta(days=day, hours=hour)
            if when.weekday() in (0,) and rng.random() < 0.15:
                continue    # 월요일 약간 감소
            category = cats[rng.choice(len(cats), p=probs)]
            txs.append(_make_tx(rng, user_id, card_id, category, when,
                                cfg["online"], cfg["variance"]))
    return txs


def _inject_recurring(rng, user_id, card_id, start_date, months, txs) -> None:
    """정기 거래: 통신요금(매월 15일±1), 구독 2종, 교통 주 5회."""
    telecom_amt = int(rng.choice([39_000, 55_000, 66_000]))
    sub_amt = [9_900, 13_500]
    for m in range(months):
        base = start_date + timedelta(days=30 * m)
        # 통신
        d = 15 + int(rng.integers(-1, 2))
        txs.append(_make_tx(rng, user_id, card_id, "SUBSCRIPTION",
                            base + timedelta(days=d, hours=10),
                            0, amount=int(telecom_amt * rng.uniform(0.97, 1.03)),
                            channel="ONLINE", merchant="SKT통신"))
        # 구독
        for j, amt in enumerate(sub_amt):
            txs.append(_make_tx(rng, user_id, card_id, "SUBSCRIPTION",
                                base + timedelta(days=1 + j, hours=3),
                                0, amount=amt, channel="ONLINE",
                                merchant="넷플릭스구독" if j == 0 else "유튜브프리미엄"))
        # 교통 (주 5회 근사: 월 20회)
        for _ in range(20):
            day = int(rng.integers(0, 30))
            txs.append(_make_tx(rng, user_id, card_id, "TRANSPORT",
                                base + timedelta(days=day, hours=float(rng.uniform(7, 21))),
                                0, amount=int(rng.choice([1400, 1500, 2500, 2800])),
                                channel="OFFLINE", merchant="서울지하철"))


def _inject_anomalies(rng, users, all_txs, anomaly_rate, hard_negative_rate,
                      start_date, end_date, travel_windows) -> None:
    """이상 시나리오(SC-01~07)와 hard negative(SC-08~12)를 주입한다."""
    from .scenarios import (make_anomaly, make_hard_negative,
                            make_hard_negative_history, ANOMALY_SCENARIOS,
                            HARD_NEG_SCENARIOS)

    # 사건 수는 사용자 수에 비례시켜 스케일에 무관하게 일정한 밀도를 유지한다.
    # (tx 비율로 잡으면 멀티거래 사건이 특정 사용자에 몰려 프로파일이 오염된다.)
    n_anom = int(len(users) * 3)      # 사용자당 평균 이상 사건 ~3건 / 12개월
    n_hard = int(len(users) * 2)      # 사용자당 평균 hard negative ~2건
    eid = 0

    for _ in range(n_anom):
        eid += 1
        u = users[int(rng.integers(0, len(users)))]
        scenario = ANOMALY_SCENARIOS[int(rng.integers(0, len(ANOMALY_SCENARIOS)))]
        when = start_date + timedelta(
            days=float(rng.uniform(30, (end_date - start_date).days - 1)))
        for tx in make_anomaly(rng, u, scenario, when):
            all_txs.append(GenTx(tx=tx, label=True, scenario=scenario, event_id=eid))

    for _ in range(n_hard):
        eid += 1
        u = users[int(rng.integers(0, len(users)))]
        scenario = HARD_NEG_SCENARIOS[int(rng.integers(0, len(HARD_NEG_SCENARIOS)))]
        when = start_date + timedelta(
            days=float(rng.uniform(30, (end_date - start_date).days - 1)))
        # 사전 이력(배경 정상)을 먼저 심어 '익숙한 맥락'을 만든다
        for tx in make_hard_negative_history(rng, u, scenario, when):
            if tx.approved_at >= start_date:
                all_txs.append(GenTx(tx=tx, label=None))
        txs = make_hard_negative(rng, u, scenario, when)
        for tx in txs:
            all_txs.append(GenTx(tx=tx, label=False, scenario=scenario, event_id=eid))
        # SC-10(여행 중 해외결제)은 여행 모드를 함께 등록해 정상 처리되게 한다
        if scenario == "SC-10" and txs:
            travel_windows.append((u["user_id"],
                                   when - timedelta(days=1),
                                   when + timedelta(days=3)))
