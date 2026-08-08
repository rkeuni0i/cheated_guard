"""도메인 모델 — dataclass 기반 (PRD 4.2 ERD).

자족형 구현이므로 ORM 대신 가벼운 dataclass를 쓰고, db.py가 SQLite 행과
상호 변환한다. 금액은 정수(KRW), 시각은 tz-aware datetime으로 다룬다.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Optional


@dataclass
class Transaction:
    """카드 승인 1건 (PRD 4.3.1)."""
    user_id: str
    card_id: str
    approval_no: str
    amount: int
    approved_at: datetime
    merchant_name: str
    mcc: str
    channel: str = "OFFLINE"        # ONLINE|OFFLINE|OVERSEAS
    tx_type: str = "APPROVAL"       # APPROVAL|CANCEL|PARTIAL_CANCEL
    installment_months: int = 0
    merchant_region: Optional[str] = None
    currency: str = "KRW"
    original_approval_no: Optional[str] = None

    # 정규화 단계에서 채워지는 파생 필드
    tx_id: Optional[int] = None
    merchant_id: Optional[int] = None
    merchant_norm: Optional[str] = None
    category: Optional[str] = None
    routine_flag: bool = False
    is_excluded: bool = False

    @property
    def hour(self) -> int:
        return self.approved_at.hour

    @property
    def dow(self) -> int:
        return self.approved_at.weekday()   # 0=월 .. 6=일


@dataclass
class Merchant:
    """가맹점 마스터 (PRD 4.3.2)."""
    name_raw: str
    name_normalized: str
    mcc: str
    category: str
    is_special: bool = False
    variance_class: str = "MID"
    merchant_id: Optional[int] = None


@dataclass
class Profile:
    """사용자 소비 프로파일 (PRD 4.3.3). stats/cat_stats/time_stats/behavior는
    JSONB 스키마(4.3)를 dict로 그대로 보관한다."""
    user_id: str
    version: int
    window_start: str
    window_end: str
    tx_count: int
    is_cold_start: bool
    stats: dict
    cat_stats: dict
    time_stats: dict
    behavior: dict
    sensitivity: str = "DEFAULT"
    peer_id: Optional[str] = None
    profile_id: Optional[int] = None

    def cat(self, category: str) -> Optional[dict]:
        return self.cat_stats.get(category)


@dataclass
class Reason:
    """사용자용 근거 1개 (PRD 6.8)."""
    code: str
    label: str
    contribution: float

    def to_dict(self) -> dict:
        return {"code": self.code, "label": self.label,
                "contribution": round(self.contribution, 3)}


@dataclass
class Assessment:
    """판정 결과 (PRD 4.3.4). 재현성을 위해 버전·피처 스냅샷 동봉."""
    tx_id: Optional[int]
    user_id: str
    score: int
    grade: str
    l1_score: int
    l2_score: int
    l3_score: int
    l3_status: str
    reasons: list[Reason]
    features: dict[str, Any]
    profile_version: int
    rule_version: str = ""
    model_version: str = ""
    override: str = "NORMAL"
    reference_source: str = "PERSONAL"
    latency_ms: int = 0
    skipped: bool = False
    skip_reason: Optional[str] = None
    assessment_id: Optional[int] = None
    created_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["reasons"] = [r.to_dict() for r in self.reasons]
        if self.created_at:
            d["created_at"] = self.created_at.isoformat()
        return d
