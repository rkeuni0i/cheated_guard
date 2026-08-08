"""SQLite 저장소 — PRD 4.3 DDL의 자족형 이식.

PostgreSQL의 파티셔닝·JSONB·GIN trigram은 SQLite에 없으므로,
JSONB는 TEXT+json, 파티셔닝은 생략, 멱등 유니크 인덱스는 그대로 유지한다.
스키마 의미(멱등성, append-only 프로파일, 재현성 버전 기록)는 보존한다.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional

from .models import Assessment, Merchant, Profile, Transaction

KST = timezone(timedelta(hours=9))

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS users (
    user_id      TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    sensitivity  TEXT NOT NULL DEFAULT 'DEFAULT',
    peer_id      TEXT,
    status       TEXT NOT NULL DEFAULT 'ACTIVE'
);

CREATE TABLE IF NOT EXISTS cards (
    card_id     TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    card_ref    TEXT NOT NULL,
    issuer      TEXT,
    status      TEXT NOT NULL DEFAULT 'ACTIVE',
    frozen_at   TEXT
);

CREATE TABLE IF NOT EXISTS merchants (
    merchant_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    name_raw        TEXT NOT NULL,
    name_normalized TEXT NOT NULL,
    mcc             TEXT,
    category        TEXT NOT NULL,
    is_special      INTEGER NOT NULL DEFAULT 0,
    variance_class  TEXT NOT NULL DEFAULT 'MID',
    first_seen_at   TEXT NOT NULL,
    UNIQUE (name_normalized, mcc)
);

CREATE TABLE IF NOT EXISTS transactions (
    tx_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id            TEXT NOT NULL,
    card_id            TEXT NOT NULL,
    merchant_id        INTEGER,
    approval_no        TEXT NOT NULL,
    amount             INTEGER NOT NULL CHECK (amount > 0),
    currency           TEXT NOT NULL DEFAULT 'KRW',
    approved_at        TEXT NOT NULL,
    mcc                TEXT,
    category           TEXT NOT NULL,
    channel            TEXT NOT NULL,
    tx_type            TEXT NOT NULL DEFAULT 'APPROVAL',
    original_tx_id     INTEGER,
    installment_months INTEGER NOT NULL DEFAULT 0,
    merchant_region    TEXT,
    routine_flag       INTEGER NOT NULL DEFAULT 0,
    is_excluded        INTEGER NOT NULL DEFAULT 0,
    ingested_at        TEXT NOT NULL,
    UNIQUE (card_id, approval_no, approved_at)
);
CREATE INDEX IF NOT EXISTS ix_tx_user_time ON transactions (user_id, approved_at DESC);
CREATE INDEX IF NOT EXISTS ix_tx_user_cat  ON transactions (user_id, category, approved_at DESC);

CREATE TABLE IF NOT EXISTS user_profiles (
    profile_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       TEXT NOT NULL,
    version       INTEGER NOT NULL,
    window_start  TEXT NOT NULL,
    window_end    TEXT NOT NULL,
    tx_count      INTEGER NOT NULL,
    is_cold_start INTEGER NOT NULL DEFAULT 0,
    stats         TEXT NOT NULL,
    cat_stats     TEXT NOT NULL,
    time_stats    TEXT NOT NULL,
    behavior      TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    UNIQUE (user_id, version)
);
CREATE INDEX IF NOT EXISTS ix_profile_latest ON user_profiles (user_id, version DESC);

CREATE TABLE IF NOT EXISTS risk_assessments (
    assessment_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    tx_id           INTEGER,
    user_id         TEXT NOT NULL,
    score           INTEGER NOT NULL,
    grade           TEXT NOT NULL,
    l1_score        INTEGER,
    l2_score        INTEGER,
    l3_score        INTEGER,
    l3_status       TEXT NOT NULL DEFAULT 'OK',
    reasons         TEXT NOT NULL,
    features        TEXT NOT NULL,
    profile_version INTEGER NOT NULL,
    rule_version    TEXT NOT NULL,
    model_version   TEXT,
    override        TEXT,
    reference_source TEXT,
    latency_ms      INTEGER,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_ra_user_time ON risk_assessments (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_ra_grade     ON risk_assessments (grade, created_at DESC);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    assessment_id INTEGER NOT NULL,
    user_id       TEXT NOT NULL,
    channel       TEXT NOT NULL,
    body          TEXT NOT NULL,
    sent_at       TEXT NOT NULL,
    responded_at  TEXT,
    response      TEXT,
    expires_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS labels (
    label_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    tx_id       INTEGER NOT NULL,
    user_id     TEXT NOT NULL,
    is_anomaly  INTEGER NOT NULL,
    source      TEXT NOT NULL,
    confidence  REAL NOT NULL DEFAULT 1.0,
    labeled_at  TEXT NOT NULL,
    UNIQUE (tx_id, source)
);

CREATE TABLE IF NOT EXISTS card_actions (
    action_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    card_id       TEXT NOT NULL,
    action_type   TEXT NOT NULL,
    triggered_by  TEXT NOT NULL,
    assessment_id INTEGER,
    status        TEXT NOT NULL,
    reason        TEXT,
    created_at    TEXT NOT NULL,
    completed_at  TEXT
);

CREATE TABLE IF NOT EXISTS audit_logs (
    log_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    actor       TEXT NOT NULL,
    action      TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_id   TEXT NOT NULL,
    payload     TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS injected_labels (
    tx_id      INTEGER PRIMARY KEY,
    is_anomaly INTEGER NOT NULL,
    scenario   TEXT,
    event_id   INTEGER
);

CREATE TABLE IF NOT EXISTS whitelist (
    user_id     TEXT NOT NULL,
    merchant_id INTEGER NOT NULL,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (user_id, merchant_id)
);

CREATE TABLE IF NOT EXISTS suppressions (
    user_id     TEXT NOT NULL,
    merchant_id INTEGER NOT NULL,
    amount_band INTEGER NOT NULL,
    until       TEXT NOT NULL,
    PRIMARY KEY (user_id, merchant_id, amount_band)
);

CREATE TABLE IF NOT EXISTS travel_mode (
    user_id  TEXT NOT NULL,
    start_at TEXT NOT NULL,
    end_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_travel_user ON travel_mode (user_id);
"""


def now_kst() -> datetime:
    return datetime.now(KST)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


class DB:
    """얇은 SQLite 래퍼 + 리포지토리 메서드."""

    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def commit(self) -> None:
        self.conn.commit()

    # ── users / cards ──────────────────────────────────────────────────
    def upsert_user(self, user_id: str, sensitivity: str = "DEFAULT",
                    peer_id: Optional[str] = None) -> None:
        self.conn.execute(
            "INSERT INTO users(user_id, created_at, sensitivity, peer_id) "
            "VALUES(?,?,?,?) ON CONFLICT(user_id) DO UPDATE SET "
            "sensitivity=excluded.sensitivity, peer_id=excluded.peer_id",
            (user_id, _iso(now_kst()), sensitivity, peer_id))

    def upsert_card(self, card_id: str, user_id: str, card_ref: str,
                    issuer: str = "SIM") -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO cards(card_id, user_id, card_ref, issuer) "
            "VALUES(?,?,?,?)", (card_id, user_id, card_ref, issuer))

    def get_user(self, user_id: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()

    def set_card_status(self, card_id: str, status: str,
                        frozen_at: Optional[datetime] = None) -> None:
        self.conn.execute(
            "UPDATE cards SET status=?, frozen_at=? WHERE card_id=?",
            (status, _iso(frozen_at) if frozen_at else None, card_id))

    def get_card(self, card_id: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM cards WHERE card_id=?", (card_id,)).fetchone()

    # ── merchants ──────────────────────────────────────────────────────
    def resolve_merchant(self, m: Merchant) -> int:
        row = self.conn.execute(
            "SELECT merchant_id FROM merchants WHERE name_normalized=? AND mcc IS ?",
            (m.name_normalized, m.mcc)).fetchone()
        if row:
            return row["merchant_id"]
        cur = self.conn.execute(
            "INSERT INTO merchants(name_raw, name_normalized, mcc, category, "
            "is_special, variance_class, first_seen_at) VALUES(?,?,?,?,?,?,?)",
            (m.name_raw, m.name_normalized, m.mcc, m.category,
             int(m.is_special), m.variance_class, _iso(now_kst())))
        return cur.lastrowid

    def set_merchant_special(self, merchant_id: int, is_special: bool,
                             variance_class: str) -> None:
        self.conn.execute(
            "UPDATE merchants SET is_special=?, variance_class=? WHERE merchant_id=?",
            (int(is_special), variance_class, merchant_id))

    def get_merchant(self, merchant_id: int) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM merchants WHERE merchant_id=?", (merchant_id,)).fetchone()

    # ── transactions ───────────────────────────────────────────────────
    def insert_transaction(self, tx: Transaction) -> Optional[int]:
        """멱등 삽입. 중복(card_id, approval_no, approved_at)이면 None 반환 (DQ-03)."""
        try:
            cur = self.conn.execute(
                "INSERT INTO transactions(user_id, card_id, merchant_id, approval_no, "
                "amount, currency, approved_at, mcc, category, channel, tx_type, "
                "original_tx_id, installment_months, merchant_region, routine_flag, "
                "is_excluded, ingested_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (tx.user_id, tx.card_id, tx.merchant_id, tx.approval_no, tx.amount,
                 tx.currency, _iso(tx.approved_at), tx.mcc, tx.category, tx.channel,
                 tx.tx_type, None, tx.installment_months, tx.merchant_region,
                 int(tx.routine_flag), int(tx.is_excluded), _iso(now_kst())))
            return cur.lastrowid
        except sqlite3.IntegrityError:
            return None

    def mark_excluded(self, tx_id: int) -> None:
        self.conn.execute(
            "UPDATE transactions SET is_excluded=1 WHERE tx_id=?", (tx_id,))

    def window_transactions(self, user_id: str, window_end: datetime,
                            window_days: int) -> list[sqlite3.Row]:
        """프로파일 학습용: window_end 이전 window_days일의 승인 거래 (누수 방지 5.1)."""
        start = window_end - timedelta(days=window_days)
        return self.conn.execute(
            "SELECT * FROM transactions WHERE user_id=? AND approved_at < ? "
            "AND approved_at >= ? AND tx_type='APPROVAL' AND is_excluded=0 "
            "ORDER BY approved_at",
            (user_id, _iso(window_end), _iso(start))).fetchall()

    def merchant_tx_count(self, user_id: str, merchant_id: int,
                          before: datetime) -> int:
        """해당 사용자의 이 가맹점 과거 거래 수 (현재 거래 제외 — before 미만)."""
        row = self.conn.execute(
            "SELECT COUNT(*) c FROM transactions WHERE user_id=? AND merchant_id=? "
            "AND approved_at < ? AND tx_type='APPROVAL'",
            (user_id, merchant_id, _iso(before))).fetchone()
        return row["c"]

    def all_merchant_amounts(self, merchant_id: int) -> list[int]:
        rows = self.conn.execute(
            "SELECT amount FROM transactions WHERE merchant_id=? AND tx_type='APPROVAL'",
            (merchant_id,)).fetchall()
        return [r["amount"] for r in rows]

    def user_ids(self) -> list[str]:
        return [r["user_id"] for r in
                self.conn.execute("SELECT user_id FROM users ORDER BY user_id")]

    def distinct_regions(self, user_id: str, before: datetime) -> set[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT merchant_region FROM transactions WHERE user_id=? "
            "AND approved_at < ? AND merchant_region IS NOT NULL",
            (user_id, _iso(before))).fetchall()
        return {r["merchant_region"] for r in rows}

    def last_tx_before(self, user_id: str, before: datetime) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM transactions WHERE user_id=? AND approved_at < ? "
            "AND tx_type='APPROVAL' ORDER BY approved_at DESC LIMIT 1",
            (user_id, _iso(before))).fetchone()

    # ── profiles (append-only) ─────────────────────────────────────────
    def next_profile_version(self, user_id: str) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(version),0)+1 v FROM user_profiles WHERE user_id=?",
            (user_id,)).fetchone()
        return row["v"]

    def save_profile(self, p: Profile) -> int:
        cur = self.conn.execute(
            "INSERT INTO user_profiles(user_id, version, window_start, window_end, "
            "tx_count, is_cold_start, stats, cat_stats, time_stats, behavior, created_at)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (p.user_id, p.version, p.window_start, p.window_end, p.tx_count,
             int(p.is_cold_start), json.dumps(p.stats), json.dumps(p.cat_stats),
             json.dumps(p.time_stats), json.dumps(p.behavior), _iso(now_kst())))
        return cur.lastrowid

    def latest_profile(self, user_id: str) -> Optional[Profile]:
        row = self.conn.execute(
            "SELECT * FROM user_profiles WHERE user_id=? ORDER BY version DESC LIMIT 1",
            (user_id,)).fetchone()
        if not row:
            return None
        u = self.get_user(user_id)
        return Profile(
            user_id=user_id, version=row["version"],
            window_start=row["window_start"], window_end=row["window_end"],
            tx_count=row["tx_count"], is_cold_start=bool(row["is_cold_start"]),
            stats=json.loads(row["stats"]), cat_stats=json.loads(row["cat_stats"]),
            time_stats=json.loads(row["time_stats"]), behavior=json.loads(row["behavior"]),
            sensitivity=u["sensitivity"] if u else "DEFAULT",
            peer_id=u["peer_id"] if u else None, profile_id=row["profile_id"])

    # ── assessments / alerts / labels / audit ──────────────────────────
    def save_assessment(self, a: Assessment) -> int:
        cur = self.conn.execute(
            "INSERT INTO risk_assessments(tx_id, user_id, score, grade, l1_score, "
            "l2_score, l3_score, l3_status, reasons, features, profile_version, "
            "rule_version, model_version, override, reference_source, latency_ms, "
            "created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (a.tx_id, a.user_id, a.score, a.grade, a.l1_score, a.l2_score,
             a.l3_score, a.l3_status, json.dumps([r.to_dict() for r in a.reasons]),
             json.dumps(a.features), a.profile_version, a.rule_version,
             a.model_version, a.override, a.reference_source, a.latency_ms,
             _iso(a.created_at or now_kst())))
        a.assessment_id = cur.lastrowid
        return cur.lastrowid

    def save_alert(self, assessment_id: int, user_id: str, channel: str,
                   body: str, expires_at: datetime) -> int:
        cur = self.conn.execute(
            "INSERT INTO alerts(assessment_id, user_id, channel, body, sent_at, "
            "expires_at) VALUES(?,?,?,?,?,?)",
            (assessment_id, user_id, channel, body, _iso(now_kst()), _iso(expires_at)))
        return cur.lastrowid

    def alerts_today(self, user_id: str, day: datetime, grades=("R1", "R2")) -> int:
        """당일 발송된 R1/R2 알림 수 (피로도 상한 7.3)."""
        start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        placeholders = ",".join("?" * len(grades))
        row = self.conn.execute(
            f"SELECT COUNT(*) c FROM alerts a JOIN risk_assessments r "
            f"ON a.assessment_id=r.assessment_id WHERE a.user_id=? "
            f"AND a.sent_at>=? AND a.sent_at<? AND r.grade IN ({placeholders})",
            (user_id, _iso(start), _iso(end), *grades)).fetchone()
        return row["c"]

    def save_label(self, tx_id: int, user_id: str, is_anomaly: bool,
                   source: str, confidence: float = 1.0) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO labels(tx_id, user_id, is_anomaly, source, "
            "confidence, labeled_at) VALUES(?,?,?,?,?,?)",
            (tx_id, user_id, int(is_anomaly), source, confidence, _iso(now_kst())))

    def save_injected_label(self, tx_id: int, is_anomaly: bool, scenario: str,
                            event_id: Optional[int] = None) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO injected_labels(tx_id, is_anomaly, scenario, "
            "event_id) VALUES(?,?,?,?)", (tx_id, int(is_anomaly), scenario, event_id))

    def injected_label(self, tx_id: int) -> Optional[bool]:
        row = self.conn.execute(
            "SELECT is_anomaly FROM injected_labels WHERE tx_id=?", (tx_id,)).fetchone()
        return bool(row["is_anomaly"]) if row else None

    def audit(self, actor: str, action: str, target_type: str, target_id: str,
              payload: Optional[dict] = None) -> None:
        self.conn.execute(
            "INSERT INTO audit_logs(actor, action, target_type, target_id, payload, "
            "created_at) VALUES(?,?,?,?,?,?)",
            (actor, action, target_type, str(target_id),
             json.dumps(payload) if payload else None, _iso(now_kst())))

    def record_card_action(self, card_id: str, action_type: str, triggered_by: str,
                           status: str, assessment_id: Optional[int] = None,
                           reason: Optional[str] = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO card_actions(card_id, action_type, triggered_by, "
            "assessment_id, status, reason, created_at, completed_at) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (card_id, action_type, triggered_by, assessment_id, status, reason,
             _iso(now_kst()), _iso(now_kst())))
        return cur.lastrowid

    # ── whitelist / suppression / travel (7.3, 6.5 override) ───────────
    def add_whitelist(self, user_id: str, merchant_id: int) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO whitelist(user_id, merchant_id, created_at) "
            "VALUES(?,?,?)", (user_id, merchant_id, _iso(now_kst())))

    def is_whitelisted(self, user_id: str, merchant_id: Optional[int]) -> bool:
        if merchant_id is None:
            return False
        return self.conn.execute(
            "SELECT 1 FROM whitelist WHERE user_id=? AND merchant_id=?",
            (user_id, merchant_id)).fetchone() is not None

    def add_suppression(self, user_id: str, merchant_id: int, amount_band: int,
                        until: datetime) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO suppressions(user_id, merchant_id, amount_band, "
            "until) VALUES(?,?,?,?)",
            (user_id, merchant_id, amount_band, _iso(until)))

    def is_suppressed(self, user_id: str, merchant_id: Optional[int],
                      amount_band: int, at: datetime) -> bool:
        if merchant_id is None:
            return False
        row = self.conn.execute(
            "SELECT until FROM suppressions WHERE user_id=? AND merchant_id=? "
            "AND amount_band=?", (user_id, merchant_id, amount_band)).fetchone()
        return bool(row) and row["until"] > _iso(at)

    def add_travel_mode(self, user_id: str, start: datetime, end: datetime) -> None:
        self.conn.execute(
            "INSERT INTO travel_mode(user_id, start_at, end_at) VALUES(?,?,?)",
            (user_id, _iso(start), _iso(end)))

    def in_travel_mode(self, user_id: str, at: datetime) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM travel_mode WHERE user_id=? AND start_at<=? AND end_at>=? "
            "LIMIT 1", (user_id, _iso(at), _iso(at))).fetchone() is not None
