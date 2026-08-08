"""판정 오케스트레이터 — assess_transaction (PRD 6.6).

L1/L2 동기 + L3 선택적(타임아웃/부재 시 제외). 융합 → 등급 → 근거 →
영속화 순. 재현성을 위해 프로파일/룰/모델 버전과 피처 스냅샷을 함께 기록한다.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

from ..config import MODEL_VERSION
from ..db import DB
from ..features.transforms import build_features
from ..features.velocity import VelocityTracker
from ..models import Assessment, Profile, Transaction
from ..profiling.peer_groups import PeerStore
from .explain import l2_reasons, l3_reasons, rank_and_humanize
from .fusion import fuse, to_grade
from .ml import L3_FEATURES, L3Model
from .rules import active_rule_version, evaluate_rules, score_l1
from .statistics import score_l2


def assess(db: DB, tx: Transaction, profile: Profile, velo: VelocityTracker,
           peer_store: Optional[PeerStore] = None,
           l3_model: Optional[L3Model] = None,
           persist: bool = True) -> Assessment:
    t0 = time.perf_counter()
    ts = VelocityTracker.ts_of(tx.approved_at)

    # 2. 피처 (벨로시티는 현재 거래 기록 전에 조회 — 누수 방지)
    velo_feats = velo.query(tx.user_id, ts)
    feats = build_features(tx, profile, velo_feats, db, peer_store)
    velo.record(tx.user_id, ts, tx.amount, tx.merchant_id)

    # 3. 3계층 스코어링
    hits = evaluate_rules(feats, tx, profile, db)
    l1_score, l1_reasons = score_l1(hits)
    l2_score, l2_axes, ref_source = score_l2(tx, profile, feats, peer_store)

    l3_score, l3_status = 0, "SKIPPED"
    if l3_model is not None:
        try:
            l3_score = l3_model.score(feats)
            l3_status = "OK"
        except Exception:
            l3_score, l3_status = 0, "TIMEOUT"

    # 4. 융합 + 등급
    score, override = fuse(l1_score, l2_score, l3_score, l3_status, tx,
                           profile, feats, l1_reasons, db)
    grade = to_grade(score, l1_reasons, profile, db, tx.approved_at)

    # 5. 근거 생성 (L1 + L2 상위축 + L3 SHAP 근사)
    reasons = list(l1_reasons)
    reasons += l2_reasons(l2_axes, tx, profile, k=2)
    if l3_status == "OK":
        reasons += l3_reasons(_l3_top_features(feats, k=3))
    reasons = rank_and_humanize(reasons, limit=3)

    latency_ms = int((time.perf_counter() - t0) * 1000)
    assessment = Assessment(
        tx_id=tx.tx_id, user_id=tx.user_id, score=score, grade=grade,
        l1_score=l1_score, l2_score=l2_score, l3_score=l3_score,
        l3_status=l3_status, reasons=reasons, features=_clean(feats),
        profile_version=profile.version, rule_version=active_rule_version(),
        model_version=MODEL_VERSION if l3_status == "OK" else "",
        override=override, reference_source=ref_source, latency_ms=latency_ms,
        created_at=tx.approved_at)

    if persist:
        db.save_assessment(assessment)
        db.commit()
    return assessment


def _l3_top_features(feats: dict, k: int = 3) -> list[tuple[str, float]]:
    """SHAP 부재 시 근사: 이상 방향으로 큰 피처를 기여도 순으로."""
    signals = {
        "merchant_novelty": feats.get("merchant_novelty", 0),
        "amt_over_p95": min(feats.get("amt_over_p95", 0) / 5, 1.0),
        "is_night": feats.get("is_night", 0),
        "is_overseas": feats.get("is_overseas", 0),
        "cat_amt_over_max": min(feats.get("cat_amt_over_max", 0) / 3, 1.0),
    }
    ranked = sorted(signals.items(), key=lambda kv: -kv[1])
    return [(f, v) for f, v in ranked[:k] if v > 0.3]


def _clean(feats: dict) -> dict:
    """JSON 직렬화 가능한 피처 스냅샷 (내부 키 제외)."""
    return {k: (round(v, 6) if isinstance(v, float) else v)
            for k, v in feats.items() if not k.startswith("_")} | \
           {"reference_source": feats.get("_reference_source", "PERSONAL")}
