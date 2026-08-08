"""이상/정상 시나리오 픽스처 SC-01~12 (PRD 13.2).

make_anomaly / make_hard_negative 는 시나리오에 맞는 Transaction 리스트를
생성한다. 백테스트의 정답 라벨로 쓰인다.
"""
from __future__ import annotations

import uuid
from datetime import timedelta
from typing import List

from spendguard.models import Transaction

ANOMALY_SCENARIOS = ["SC-01", "SC-02", "SC-03", "SC-04", "SC-05", "SC-06", "SC-07"]
HARD_NEG_SCENARIOS = ["SC-08", "SC-09", "SC-10", "SC-11", "SC-12"]


def _tx(u, when, amount, merchant, mcc, channel="OFFLINE",
        region="서울 강남구", installment=0) -> Transaction:
    return Transaction(
        user_id=u["user_id"], card_id=u["card_id"],
        approval_no=uuid.uuid4().hex[:12], amount=int(amount),
        approved_at=when, merchant_name=merchant, mcc=mcc, channel=channel,
        merchant_region=region if channel != "ONLINE" else None,
        installment_months=installment)


def make_anomaly(rng, u, scenario, when) -> List[Transaction]:
    """라벨=True 이상 거래 생성."""
    if scenario == "SC-01":     # 단건 고액 (대표 사례)
        return [_tx(u, when, rng.integers(300_000, 500_000), "컴퓨존", "5732",
                    channel="ONLINE")]

    if scenario == "SC-02":     # 카드 도난 후 연속 결제 (1시간 내 5건, 다른 가맹점)
        merchants = [("롯데백화점", "5311"), ("신세계免税", "5309"),
                     ("하이마트", "5732"), ("애플스토어", "5732"), ("현대백화점", "5311")]
        out = []
        t = when
        for name, mcc in merchants:
            out.append(_tx(u, t, rng.integers(300_000, 500_000), name, mcc,
                           channel="OFFLINE"))
            t = t + timedelta(minutes=int(rng.integers(5, 12)))
        return out

    if scenario == "SC-03":     # 소액 테스트 후 고액
        return [_tx(u, when, rng.integers(1_000, 5_000), "온라인테스트샵", "5999",
                    channel="ONLINE"),
                _tx(u, when + timedelta(minutes=8),
                    rng.integers(400_000, 700_000), "럭셔리몰", "5944",
                    channel="ONLINE")]

    if scenario == "SC-04":     # 심야 해외 결제
        night = when.replace(hour=3, minute=int(rng.integers(0, 59)))
        return [_tx(u, night, rng.integers(400_000, 900_000), "AMZN GLOBAL",
                    "5999", channel="OVERSEAS", region=None)]

    if scenario == "SC-05":     # 서서히 증가하는 유출 (동일 가맹점, 금액 점증)
        # 초반 소액은 개별 탐지가 어렵고 후반부에 누적·금액으로 잡히는 패턴.
        amounts = [40_000, 70_000, 110_000, 170_000, 250_000, 350_000]
        out = []
        for i, amt in enumerate(amounts):
            out.append(_tx(u, when + timedelta(days=i * 3, hours=2),
                           amt, "해외구독서비스", "5968", channel="ONLINE",
                           region=None))
        return out

    if scenario == "SC-06":     # 미경험 카테고리 고액 (귀금속)
        return [_tx(u, when, rng.integers(700_000, 1_200_000), "골드바코리아",
                    "5944", channel="OFFLINE")]

    if scenario == "SC-07":     # 장기 미사용 후 고액 (인프로세스 근사)
        return [_tx(u, when, rng.integers(500_000, 800_000), "가전랜드", "5732",
                    channel="OFFLINE")]

    return []


def make_hard_negative(rng, u, scenario, when) -> List[Transaction]:
    """라벨=False — 이상해 보이지만 정상인 거래."""
    if scenario == "SC-08":     # 월급날 정기 고액 (routine 성격)
        payday = when.replace(day=min(25, 28))
        return [_tx(u, payday, 450_000, "보험료자동납부", "6300",
                    channel="ONLINE")]

    if scenario == "SC-09":     # 올리브영 세일 대량구매 (단골 특수 가맹점)
        return [_tx(u, when, rng.integers(200_000, 300_000), "올리브영", "5977",
                    channel="OFFLINE")]

    if scenario == "SC-10":     # 여행 중 해외 결제 (여행 모드 가정)
        out = []
        for i in range(3):
            out.append(_tx(u, when + timedelta(hours=i * 3),
                           rng.integers(50_000, 150_000), "TOKYO SHOP", "5814",
                           channel="OVERSEAS", region=None))
        return out

    if scenario == "SC-11":     # 명절 선물 (백화점 고액)
        return [_tx(u, when, rng.integers(200_000, 400_000), "현대백화점", "5311",
                    channel="OFFLINE")]

    if scenario == "SC-12":     # 할부 가전 (단골 매장, 12개월)
        return [_tx(u, when, 1_500_000, "삼성디지털프라자", "5732",
                    channel="OFFLINE", installment=12)]

    return []


def make_hard_negative_history(rng, u, scenario, when) -> List[Transaction]:
    """hard negative의 '평소 맥락'을 만드는 사전 이력 (라벨=None, 배경 정상).

    이 이력이 있어야 SC-08(정기 보험료)·SC-12(단골 매장 할부) 등이 실제로
    '익숙한 거래'로 보이고, 시스템이 정상 처리할 근거를 갖는다.
    """
    out = []
    if scenario == "SC-08":     # 매월 반복되는 보험료 (routine)
        for m in range(1, 13):
            out.append(_tx(u, when - timedelta(days=30 * m),
                           int(450_000 * rng.uniform(0.98, 1.02)),
                           "보험료자동납부", "6300", channel="ONLINE"))
    elif scenario == "SC-09":   # 올리브영 단골 (특수 가맹점, 금액 다양)
        for m in range(14):
            out.append(_tx(u, when - timedelta(days=int(rng.uniform(5, 180))),
                           int(rng.integers(8_000, 250_000)), "올리브영", "5977",
                           channel="OFFLINE"))
    elif scenario == "SC-11":   # 백화점 연 2회 방문 이력
        for m in range(3):
            out.append(_tx(u, when - timedelta(days=int(rng.uniform(60, 330))),
                           int(rng.integers(150_000, 400_000)), "현대백화점", "5311",
                           channel="OFFLINE"))
    elif scenario == "SC-12":   # 삼성디지털프라자 단골 이력
        for m in range(12):
            out.append(_tx(u, when - timedelta(days=int(rng.uniform(10, 200))),
                           int(rng.integers(30_000, 300_000)),
                           "삼성디지털프라자", "5732", channel="OFFLINE"))
    return out


# 기대 등급 (PRD 13.2) — 백테스트 리포트 대조용
EXPECTED_GRADE = {
    "SC-01": {"R3"}, "SC-02": {"R4"}, "SC-03": {"R4"}, "SC-04": {"R4"},
    "SC-05": {"R2", "R3"}, "SC-06": {"R3", "R4"}, "SC-07": {"R3"},
    "SC-08": {"R0", "R1"}, "SC-09": {"R0", "R1"}, "SC-10": {"R1", "R2"},
    "SC-11": {"R1", "R2"}, "SC-12": {"R1", "R2"},
}
