"""벨로시티 슬라이딩 윈도우 (PRD 5.2).

PRD는 Redis ZSET을 쓰지만 자족형이므로 인메모리 deque로 동일 의미를 구현한다.
누수 방지 규칙(PRD 5.2 주석)을 지킨다: 조회는 현재 거래 기록 '이전'에 한다.
"""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime


class VelocityTracker:
    """사용자별 최근 24시간 거래를 (ts, amount, merchant_id)로 보관."""

    def __init__(self):
        self._events: dict[str, deque] = defaultdict(deque)

    def _evict(self, user_id: str, now_ts: float) -> None:
        dq = self._events[user_id]
        while dq and dq[0][0] < now_ts - 86400:
            dq.popleft()

    def query(self, user_id: str, ts: float) -> dict:
        """현재 거래를 기록하기 전에 과거 벨로시티를 조회한다 (누수 방지)."""
        self._evict(user_id, ts)
        dq = self._events[user_id]
        cnt_1h = cnt_24h = 0
        sum_1h = sum_24h = 0
        mset_1h = set()
        for (t, amt, mid) in dq:
            if t >= ts - 86400:
                cnt_24h += 1
                sum_24h += amt
            if t >= ts - 3600:
                cnt_1h += 1
                sum_1h += amt
                mset_1h.add(mid)
        return {
            "tx_count_1h": cnt_1h, "tx_count_24h": cnt_24h,
            "tx_sum_1h": sum_1h, "tx_sum_24h": sum_24h,
            "distinct_merchants_1h": len(mset_1h),
        }

    def record(self, user_id: str, ts: float, amount: int, merchant_id) -> None:
        """조회 이후 현재 거래를 기록한다."""
        self._events[user_id].append((ts, amount, merchant_id))

    @staticmethod
    def ts_of(dt: datetime) -> float:
        return dt.timestamp()
