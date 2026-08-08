"""L3 ML 모델 — IsolationForest (PRD 6.4 Stage 1).

라벨이 거의 없으므로 비지도 우선. 원점수는 스케일이 없어 학습셋 분위수로
캘리브레이션해 0~100으로 만든다. ML은 선택적 구성요소이며, 모델이 없거나
타임아웃이면 L1+L2만으로 판정한다(PRD 3.2 원칙 ①, 6.5).
"""
from __future__ import annotations

import numpy as np

from ..config import MODEL_VERSION

L3_FEATURES = [
    "amt_over_median", "amt_over_p95", "log_robust_z", "amt_percentile",
    "cat_amt_over_p95", "cat_amt_over_max", "merchant_novelty",
    "merchant_tx_count", "cat_novelty", "hour_rarity", "dow_rarity",
    "is_night", "secs_since_prev_tx", "tx_count_1h", "tx_count_24h",
    "sum_24h_over_daily_avg", "distinct_merchants_1h", "is_online",
    "online_rate_deviation", "is_overseas", "region_novelty",
    "is_installment", "routine_flag", "mtd_over_monthly_ewma",
    "budget_burn_deviation",
]


class L3Model:
    """학습된 IsolationForest + 분위수 변환 + 캘리브레이터 묶음."""

    def __init__(self, model, qt, calib_scores: np.ndarray):
        self.model = model
        self.qt = qt
        self.calib = np.sort(calib_scores)
        self.version = MODEL_VERSION

    def _vec(self, feats: dict) -> np.ndarray:
        return np.array([[float(feats.get(f, 0.0)) for f in L3_FEATURES]])

    def score(self, feats: dict) -> int:
        """0~100 이상 점수. 원점수를 학습 분포 분위수로 캘리브레이션."""
        x = self.qt.transform(self._vec(feats))
        raw = -self.model.score_samples(x)[0]
        pct = np.searchsorted(self.calib, raw) / max(len(self.calib), 1)
        return int(round(pct * 100))


def train_stage1(feature_rows: list[dict], contamination: float = 0.01) -> "L3Model":
    """정상 거래 피처로 IsolationForest 학습 (PRD 6.4)."""
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import QuantileTransformer

    X = np.array([[float(r.get(f, 0.0)) for f in L3_FEATURES] for r in feature_rows])
    n_q = min(1000, max(10, len(X)))
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=n_q,
                             random_state=42)
    Xt = qt.fit_transform(X)
    model = IsolationForest(
        n_estimators=300, max_samples=min(256, len(X)),
        contamination=contamination, max_features=0.8, bootstrap=False,
        random_state=42, n_jobs=-1)
    model.fit(Xt)
    calib = -model.score_samples(Xt)
    return L3Model(model, qt, calib)
