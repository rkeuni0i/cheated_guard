"""동질 집단(Peer Group) 폴백 (PRD 5.4).

민감 정보(나이·성별·소득) 없이 행동 기반으로만 군집화한다.
콜드스타트/희소 카테고리일 때 개인 통계 대신 peer 분포를 빌려온다.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..config import CAT_FALLBACK_MIN_N, BLEND_FULL_TX
from ..models import Profile


def _bucket_spend(monthly: float) -> str:
    for lim, name in [(500_000, "s0"), (1_000_000, "s1"),
                      (2_000_000, "s2"), (4_000_000, "s3")]:
        if monthly <= lim:
            return name
    return "s4"


def _bucket_freq(weekly_ewma: float) -> str:
    monthly = weekly_ewma * 4.3
    if monthly <= 20:
        return "f0"
    if monthly <= 60:
        return "f1"
    return "f2"


def _bucket_online(ratio: float) -> str:
    if ratio <= 0.2:
        return "o0"
    if ratio <= 0.5:
        return "o1"
    return "o2"


def peer_id_from_profile(stats: dict, behavior: dict) -> str:
    """프로파일 통계로부터 peer group 키를 도출 (최대 5×3×3=45 그룹)."""
    monthly = stats.get("monthly_sum_ewma") or stats.get("sum", 0)
    return "|".join([
        _bucket_spend(monthly),
        _bucket_freq(behavior.get("weekly_tx_count_ewma", 0)),
        _bucket_online(behavior.get("online_ratio", 0)),
    ])


class PeerStore:
    """peer group × 카테고리 → 참조 통계. 전체 프로파일에서 사전 집계."""

    def __init__(self):
        self.by_group: dict[str, dict[str, dict]] = {}
        self.global_cat: dict[str, dict] = {}

    @classmethod
    def build(cls, profiles: list[Profile]) -> "PeerStore":
        store = cls()
        # peer group × category 별 금액 표본을 median/p95/max 대표값으로 모은다
        grp_cat_meds: dict[tuple, list] = defaultdict(list)
        glob_cat_meds: dict[str, list] = defaultdict(list)
        for p in profiles:
            pid = peer_id_from_profile(p.stats, p.behavior)
            for cat, cs in p.cat_stats.items():
                if cs.get("n", 0) >= 1:
                    grp_cat_meds[(pid, cat)].append(cs)
                    glob_cat_meds[cat].append(cs)

        def _agg(cslist: list[dict]) -> dict:
            med = float(np.median([c["median"] for c in cslist]))
            p95 = float(np.median([c.get("p95", c["median"]) for c in cslist]))
            mx = float(np.max([c.get("max", c["median"]) for c in cslist]))
            n = int(np.sum([c.get("n", 0) for c in cslist]))
            return {"median": med, "p95": p95, "max": mx, "n": n}

        groups: dict[str, dict[str, dict]] = defaultdict(dict)
        for (pid, cat), cslist in grp_cat_meds.items():
            groups[pid][cat] = _agg(cslist)
        store.by_group = dict(groups)
        store.global_cat = {cat: _agg(cslist) for cat, cslist in glob_cat_meds.items()}
        return store

    def peer_stats(self, peer_id: str, category: str) -> dict:
        grp = self.by_group.get(peer_id, {})
        if category in grp:
            return grp[category]
        # 그룹에 해당 카테고리가 없으면 전역 폴백
        return self.global_cat.get(category, {"median": 30_000, "p95": 100_000,
                                              "max": 200_000, "n": 0})


def _blend(personal: dict, peer: dict, w: float) -> dict:
    """개인·peer 통계 선형 혼합 (w=개인 비중)."""
    def mix(k, default):
        pv = personal.get(k, default)
        qv = peer.get(k, default)
        return w * pv + (1 - w) * qv
    return {
        "median": mix("median", 30_000),
        "p95": mix("p95", 100_000),
        "max": mix("max", 200_000),
        "n": personal.get("n", 0),
    }


def get_reference_stats(profile: Profile, category: str,
                        peer_store: PeerStore | None) -> tuple[dict, str]:
    """3단 폴백 체인 (PRD 5.4): PERSONAL → BLENDED → PEER."""
    cat = profile.cat_stats.get(category)
    if cat and cat.get("n", 0) >= CAT_FALLBACK_MIN_N:
        return cat, "PERSONAL"

    if peer_store is None:
        # peer store 없으면 개인 전체 통계라도 사용
        return (cat or profile.stats), "PERSONAL"

    peer = peer_store.peer_stats(profile.peer_id or "", category)
    if profile.tx_count >= 30:
        w = min(profile.tx_count / BLEND_FULL_TX, 1.0)
        base = cat or {"median": profile.stats.get("median", 30_000),
                       "p95": profile.stats.get("p95", 100_000),
                       "max": profile.stats.get("max", 200_000),
                       "n": cat.get("n", 0) if cat else 0}
        return _blend(base, peer, w), "BLENDED"
    return peer, "PEER"
