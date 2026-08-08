"""탐지 성능 지표 (PRD 1.3, 10.2).

불균형 데이터이므로 Accuracy는 쓰지 않는다(PRD 1.3 경고). PR-AUC,
Recall@FPR, Precision@k, Lift를 쓴다.
"""
from __future__ import annotations

import numpy as np


def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Precision-Recall 곡선 아래 면적 (average precision)."""
    from sklearn.metrics import average_precision_score
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(set(y_true.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray,
                  target_fpr: float = 0.005) -> tuple[float, float]:
    """FPR ≤ target 조건에서 달성 가능한 최대 Recall과 임계값."""
    order = np.argsort(-scores)
    y = y_true[order]
    P = y.sum()
    N = len(y) - P
    if P == 0 or N == 0:
        return float("nan"), float("nan")
    tp = fp = 0
    best_recall = 0.0
    thr = scores[order][0]
    for i in range(len(y)):
        if y[i] == 1:
            tp += 1
        else:
            fp += 1
        if fp / N <= target_fpr:
            best_recall = tp / P
            thr = scores[order][i]
    return best_recall, float(thr)


def precision_at_k(y_true: np.ndarray, scores: np.ndarray,
                   frac: float = 0.005) -> float:
    """스코어 상위 frac 비율의 정밀도."""
    k = max(1, int(len(scores) * frac))
    idx = np.argsort(-scores)[:k]
    return float(y_true[idx].mean())


def lift_at(y_true: np.ndarray, scores: np.ndarray, frac: float = 0.01) -> float:
    base = y_true.mean()
    if base == 0:
        return float("nan")
    return precision_at_k(y_true, scores, frac) / base


def evaluate(y_true: list[int], scores: list[float]) -> dict:
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    rec, thr = recall_at_fpr(y, s, 0.005)
    return {
        "n": int(len(y)), "positives": int(y.sum()),
        "pr_auc": pr_auc(y, s), "roc_auc": roc_auc(y, s),
        "recall_at_fpr_0.5%": rec, "threshold_at_fpr": thr,
        "precision_at_top_0.5%": precision_at_k(y, s, 0.005),
        "lift_at_1%": lift_at(y, s, 0.01),
    }
