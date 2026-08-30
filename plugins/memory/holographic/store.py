"""
SQLite-backed fact store with entity resolution and trust scoring.
Single-user Hermes memory store plugin.
"""

import os
import re
import sqlite3
import threading
from pathlib import Path

try:
    from . import holographic as hrr
except ImportError:
    import holographic as hrr  # type: ignore[no-redef]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content         TEXT NOT NULL UNIQUE,
    category        TEXT DEFAULT 'general',
    tags            TEXT DEFAULT '',
    trust_score     REAL DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    helpful_count   INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hrr_vector      BLOB,
    msg_id          INTEGER
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    entity_type TEXT DEFAULT 'unknown',
    aliases     TEXT DEFAULT '',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fact_entities (
    fact_id   INTEGER REFERENCES facts(fact_id),
    entity_id INTEGER REFERENCES entities(entity_id),
    PRIMARY KEY (fact_id, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_facts_trust    ON facts(trust_score DESC);
CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
CREATE INDEX IF NOT EXISTS idx_entities_name  ON entities(name);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
    USING fts5(content, tags, tokenize='trigram');
CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, COALESCE(new.content,''), COALESCE(new.tags,''));
END;

CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    DELETE FROM facts_fts WHERE rowid = old.fact_id;
END;

CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
    DELETE FROM facts_fts WHERE rowid = old.fact_id;
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, COALESCE(new.content,''), COALESCE(new.tags,''));
END;

CREATE TABLE IF NOT EXISTS memory_banks (
    bank_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    bank_name  TEXT NOT NULL UNIQUE,
    vector     BLOB NOT NULL,
    dim        INTEGER NOT NULL,
    fact_count INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# Trust adjustment constants
_HELPFUL_DELTA   =  0.05
_UNHELPFUL_DELTA = -0.10
_TRUST_MIN       =  0.0
_TRUST_MAX       =  1.0

# Entity extraction patterns
_RE_CAPITALIZED  = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
_RE_DOUBLE_QUOTE = re.compile(r'"([^"]+)"')
_RE_SINGLE_QUOTE = re.compile(r"'([^']+)'")
_RE_AKA          = re.compile(
    r'(\w+(?:\s+\w+)*)\s+(?:aka|also known as)\s+(\w+(?:\s+\w+)*)',
    re.IGNORECASE,
)

# Generic business/grammar words that must never become entities.
# Curated from the 2026-08-30 entity-graph audit of the production store:
# the top ~60 2-char CJK entities by link count were ALL generic words
# (结论 593, 新增 518, 止损 377, …) while real 2-char companies sat far
# below (腾讯 #75, 智谱 #176). Frequency-based deletion would kill the
# real ones; this list is the audited alternative.
_ENTITY_STOPWORDS = frozenset({
    # audit top offenders (link-count desc)
    "结论", "新增", "止损", "营收", "净利", "字段", "修复", "导致", "亿元",
    "减仓", "而非", "分钟", "返回", "改为", "现价", "调用", "标的", "通过",
    "数据", "模式", "待办", "浮盈", "当前", "不是", "参数", "信号", "买入",
    "验证", "量比", "使用", "任务", "算力", "板块", "市值", "权重", "评分",
    "全部", "涨停", "阈值", "覆盖", "更新", "文件", "函数", "仓位", "成本",
    "触发", "格式", "删除", "路径", "确认", "新闻", "完成", "决策", "问题",
    "方案",
    # grammar glue (same shape as the offenders; added at curation)
    "关于", "说明", "显示", "我们", "当时", "投资", "公司", "业务", "讨论",
    "这个", "那个", "还是", "以及", "今天", "昨天", "明天", "如果", "因为",
    "但是", "然后", "所以", "并且", "或者", "一个", "这种", "每个",
    "可以", "需要", "应该", "就是", "只有", "还有", "已经", "正在", "状态",
    "继续", "开始", "结束", "进行", "出来", "进去", "上去", "下来",
})

# Single grammar glue chars that segment long CJK runs. Chinese has no
# spaces; a 9+ char run exceeds the entity cap and is dropped WHOLE,
# swallowing mid-sentence names (the original 百合花 recall lesson).
# Splitting on function words recovers 智谱 from 今天讨论智谱的估值.
# Deliberately EXCLUDES chars common inside real names (用/有/无/为/新/大).
_ENTITY_GLUE_CHARS = frozenset("的了吗呢吧啊与和或者是在对于从被把给向着过将已也都还又再很太最更至让不")

# Segmentation pattern: stopword substrings (longest-first) + glue chars,
# used to split over-long CJK runs at function-word boundaries.
_ENTITY_SPLIT_RE = re.compile(
    "|".join(map(re.escape, sorted(_ENTITY_STOPWORDS | _ENTITY_GLUE_CHARS,
                                   key=len, reverse=True))))

# [L|yymmdd] / [L|yymmdd|主体] distillation headers: keep the subject
# entity, drop the date (18.5% of entity links in the 2026-08-30 audit
# were date stamps eaten by the ticker regex).
_RE_HEADER = re.compile(r"^\s*\[L\|(\d{6})(?:\|([^\]]*))?\]\s*")
# CJK runs (Unified Ideographs + Ext-A), later filtered to 2-8 chars
_RE_CJK_RUN      = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]+')
# CamelCase tech tokens with an internal case boundary: MiniMax/OpenClaw/DuckDB.
# Lookarounds instead of \b: CJK chars are also \w in Python re, so \b never
# fires between 对 and M in "对MiniMax的关键判断" — unspaced mixed runs need
# the boundary asserted only against OTHER latin letters.
_RE_CAMEL_TOKEN  = re.compile(
    r'(?<![A-Za-z])[A-Za-z][a-z]+[A-Z][A-Za-z]+(?![A-Za-z])')
# Ticker-style codes: bare 6-digit A-share codes (688507 / 002222 /
# 301421 / 600895 / 000063 …), exchange-prefixed (sh600895), and
# suffix forms (2513.HK / 002222.SZ). Bare 6-digit DATE stamps (260628
# = yymmdd from the [L|…] header, or 251231) must NOT match: anchor the
# bare form to real A-share board prefixes (60/68/00/30/8/4/9).
_RE_TICKER       = re.compile(
    r'\b(?:(?:6[08]\d{4}|00\d{4}|30\d{4}|[849]\d{5})'  # bare A-share
    r'|[A-Za-z]{2}\d{6}'                               # sh600895 style
    r'|\d{3,5}\.(?:HK|SZ|SH))\b'                       # 2513.HK style
)


def _clamp_trust(value: float) -> float:
    return max(_TRUST_MIN, min(_TRUST_MAX, value))


class MemoryStore:
    """SQLite-backed fact store with entity resolution and trust scoring."""

    # --- Process-wide shared connection registry -------------------------
    # SQLite permits only one writer at a time. Each MemoryStore instance used
    # to open its own connection guarded by its own RLock, so the several
    # providers that coexist in one process (the main agent plus every
    # delegate_task subagent) raced as independent WAL writers. Combined with
    # writes that were not rolled back on error, one connection could leave an
    # open write transaction that pinned the write lock and made every other
    # connection's write fail with "database is locked" for the full busy
    # timeout. All instances for the same database now share ONE connection and
    # ONE re-entrant lock, so access is fully serialized and cross-connection
    # contention is impossible. The shared connection is refcounted, so closing
    # one instance never tears the connection out from under a live sibling.
    _shared: dict = {}
    _shared_guard = threading.Lock()

    def __init__(
        self,
        db_path: "str | Path | None" = None,
        default_trust: float = 0.5,
        hrr_dim: int = 1024,
        embedding_client=None,
        _vec_cache_max_facts: int = 50000,
    ) -> None:
        if db_path is None:
            from hermes_constants import get_hermes_home
            db_path = str(get_hermes_home() / "memory_store.db")
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.default_trust = _clamp_trust(default_trust)
        self.hrr_dim = hrr_dim
        self._hrr_available = hrr._HAS_NUMPY
        # Optional semantic leg. Any object exposing .available and
        # .embed(texts) works; None keeps the store purely lexical.
        self._embed_client = embedding_client

        # In-process embedding matrix cache (Fix 2026-08-30): the semantic
        # leg used to re-fetch every embedding blob per query (O(n) SQL +
        # Python unpack, 3-7s at 27k facts). Cache the normalized matrix in
        # RAM (50k facts x 512d float32 = ~100MB cap) and serve queries as
        # a single matmul. Invalidated by any fact mutation.
        self._vec_cache_max_facts = _vec_cache_max_facts
        self._vec_matrix = None          # np.ndarray (n, d), L2-normalized
        self._vec_ids = None             # list[int] fact_ids aligned to rows
        self._vec_cache_stats = {"builds": 0, "hits": 0, "capped": False}
        # alias for readability at call sites
        self._vec_stats = self._vec_cache_stats

        # Acquire (or open) the process-wide shared connection for this DB.
        # resolve() (not just expanduser) so symlinked/relative paths to the
        # same file share ONE connection instead of silently reintroducing
        # the multi-writer contention this registry exists to prevent.
        try:
            self._key = str(self.db_path.resolve())
        except OSError:
            self._key = str(self.db_path)
        with MemoryStore._shared_guard:
            entry = MemoryStore._shared.get(self._key)
            if entry is None:
                conn = sqlite3.connect(
                    self._key,
                    check_same_thread=False,
                    timeout=10.0,
                    # Autocommit: every statement is its own transaction, so a
                    # write that raises mid-method can never leave a dangling
                    # transaction (and its write lock) open. The explicit
                    # commit() calls below become harmless no-ops.
                    isolation_level=None,
                )
                conn.row_factory = sqlite3.Row
                entry = {"conn": conn, "lock": threading.RLock(), "refs": 0, "ready": False}
                MemoryStore._shared[self._key] = entry
            entry["refs"] += 1
            self._entry = entry
            self._conn = entry["conn"]
            self._lock = entry["lock"]

        # Initialise the schema once per shared connection.
        with self._lock:
            if not self._entry["ready"]:
                self._init_db()
                self._entry["ready"] = True

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create tables, indexes, and triggers if they do not exist. Enable WAL mode."""
        # Use the shared WAL-fallback helper so memory_store.db degrades
        # gracefully on NFS/SMB/FUSE-mounted HERMES_HOME (same issue as
        # state.db / kanban.db — see hermes_state._WAL_INCOMPAT_MARKERS).
        from hermes_state import apply_wal_with_fallback
        apply_wal_with_fallback(self._conn, db_label="memory_store.db (holographic)")
        self._conn.executescript(_SCHEMA)
        # Migrate: add hrr_vector column if missing (safe for existing databases)
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(facts)").fetchall()}
        if "hrr_vector" not in columns:
            self._conn.execute("ALTER TABLE facts ADD COLUMN hrr_vector BLOB")
        # Migrate: add msg_id (provenance) column if missing — the 2026-08-29
        # double-distill incident's blind spot: facts could not be traced
        # back to source messages.
        if "msg_id" not in columns:
            self._conn.execute("ALTER TABLE facts ADD COLUMN msg_id INTEGER")
        if "embedding" not in columns:
            # Optional semantic leg (512-dim float32 BLOB, ~2KB/fact).
            # Populated asynchronously by _compute_embedding when an
            # embedding client is configured; NULL for legacy rows.
            self._conn.execute("ALTER TABLE facts ADD COLUMN embedding BLOB")
        # Migrate: legacy facts_fts used external-content mode (content=facts)
        # with the default unicode61 tokenizer — CJK text was unindexed and
        # the external-content table can silently return 0 rows under
        # trigram. Detect the legacy shape and rebuild as a standalone
        # trigram table repopulated from facts.
        fts_sql_row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'facts_fts'"
        ).fetchone()
        if fts_sql_row and "trigram" not in (fts_sql_row[0] or ""):
            self._conn.executescript(
                """
                DROP TRIGGER IF EXISTS facts_ai;
                DROP TRIGGER IF EXISTS facts_ad;
                DROP TRIGGER IF EXISTS facts_au;
                DROP TABLE IF EXISTS facts_fts;
                """
            )
            self._conn.executescript(_SCHEMA)
            self._conn.execute(
                """
                INSERT INTO facts_fts(rowid, content, tags)
                SELECT fact_id, COALESCE(content, ''), COALESCE(tags, '')
                FROM facts
                """
            )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_fact(
        self,
        content: str,
        category: str = "general",
        tags: str = "",
        msg_id: "int | None" = None,
    ) -> int:
        """Insert a fact and return its fact_id.

        Deduplicates by content (UNIQUE constraint). On duplicate, returns
        the existing fact_id; if the duplicate call carries msg_id and the
        existing row lacks it, the provenance is backfilled (first
        provenance wins — never overwritten). Extracts entities from
        the content and links them to the fact.
        """
        with self._lock:
            content = content.strip()
            if not content:
                raise ValueError("content must not be empty")

            try:
                cur = self._conn.execute(
                    """
                    INSERT INTO facts (content, category, tags, trust_score, msg_id)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (content, category, tags, self.default_trust, msg_id),
                )
                self._conn.commit()
                fact_id: int = cur.lastrowid  # type: ignore[assignment]
            except sqlite3.IntegrityError:
                # Duplicate content — return existing id; backfill missing
                # provenance (first wins, never overwritten).
                row = self._conn.execute(
                    "SELECT fact_id, msg_id FROM facts WHERE content = ?", (content,)
                ).fetchone()
                if msg_id is not None and row["msg_id"] is None:
                    self._conn.execute(
                        "UPDATE facts SET msg_id = ? WHERE fact_id = ?",
                        (msg_id, row["fact_id"]),
                    )
                    self._conn.commit()
                return int(row["fact_id"])

            # Entity extraction and linking
            self._invalidate_vec_cache()
            for name in self._extract_entities(content):
                entity_id = self._resolve_entity(name)
                self._link_fact_entity(fact_id, entity_id)

            # Compute HRR vector after entity linking
            self._compute_hrr_vector(fact_id, content)
            # Semantic vector (no-op without an embedding client)
            self._compute_embedding(fact_id, content)
            self._rebuild_bank(category)

            return fact_id

    def search_facts(
        self,
        query: str,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
    ) -> list[dict]:
        """Full-text search over facts using FTS5.

        Returns a list of fact dicts ordered by FTS5 rank, then trust_score
        descending. Also increments retrieval_count for matched facts.
        """
        with self._lock:
            query = query.strip()
            if not query:
                return []

            # FTS5 AND-joins tokens by default, which zeroes out recall on
            # natural-language queries. Reuse the retriever's sanitizer
            # (stopword drop + OR-join content tokens). Imported lazily to
            # avoid a store->retrieval import cycle.
            from plugins.memory.holographic.retrieval import FactRetriever

            match_query = FactRetriever._sanitize_fts_query(query)
            params: list = [match_query, min_trust]
            category_clause = ""
            if category is not None:
                category_clause = "AND f.category = ?"
                params.append(category)
            params.append(limit)

            sql = f"""
                SELECT f.fact_id, f.content, f.category, f.tags,
                       f.trust_score, f.retrieval_count, f.helpful_count,
                       f.created_at, f.updated_at
                FROM facts f
                JOIN facts_fts fts ON fts.rowid = f.fact_id
                WHERE facts_fts MATCH ?
                  AND f.trust_score >= ?
                  {category_clause}
                ORDER BY fts.rank, f.trust_score DESC
                LIMIT ?
            """

            rows = self._conn.execute(sql, params).fetchall()
            results = [self._row_to_dict(r) for r in rows]

            if results:
                ids = [r["fact_id"] for r in results]
                placeholders = ",".join("?" * len(ids))
                self._conn.execute(
                    f"UPDATE facts SET retrieval_count = retrieval_count + 1 WHERE fact_id IN ({placeholders})",
                    ids,
                )
                self._conn.commit()

            return results

    def update_fact(
        self,
        fact_id: int,
        content: str | None = None,
        trust_delta: float | None = None,
        tags: str | None = None,
        category: str | None = None,
    ) -> bool:
        """Partially update a fact. Trust is clamped to [0, 1].

        Returns True if the row existed, False otherwise.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT fact_id, trust_score FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()
            if row is None:
                return False

            assignments: list[str] = ["updated_at = CURRENT_TIMESTAMP"]
            params: list = []

            if content is not None:
                assignments.append("content = ?")
                params.append(content.strip())
            if tags is not None:
                assignments.append("tags = ?")
                params.append(tags)
            if category is not None:
                assignments.append("category = ?")
                params.append(category)
            if trust_delta is not None:
                new_trust = _clamp_trust(row["trust_score"] + trust_delta)
                assignments.append("trust_score = ?")
                params.append(new_trust)

            params.append(fact_id)
            self._conn.execute(
                f"UPDATE facts SET {', '.join(assignments)} WHERE fact_id = ?",
                params,
            )
            self._conn.commit()
            if content is not None:
                # Content change → new embedding will be computed below and
                # the matrix row set changes shape — invalidate the cache.
                self._invalidate_vec_cache()

            # If content changed, re-extract entities
            if content is not None:
                self._conn.execute(
                    "DELETE FROM fact_entities WHERE fact_id = ?", (fact_id,)
                )
                for name in self._extract_entities(content):
                    entity_id = self._resolve_entity(name)
                    self._link_fact_entity(fact_id, entity_id)
                self._conn.commit()

            # Recompute HRR vector if content changed
            if content is not None:
                self._compute_hrr_vector(fact_id, content)
                # Recompute embedding too — a content change with a stale
                # vector silently poisons the semantic leg (the row keeps
                # matching the OLD text forever).
                try:
                    self._compute_embedding(fact_id, content)
                except Exception:
                    pass
            # Rebuild bank for relevant category
            cat = category or self._conn.execute(
                "SELECT category FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()["category"]
            self._rebuild_bank(cat)

            return True

    def remove_fact(self, fact_id: int) -> bool:
        """Delete a fact and its entity links. Returns True if the row existed."""
        with self._lock:
            row = self._conn.execute(
                "SELECT fact_id, category FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()
            if row is None:
                return False

            self._conn.execute(
                "DELETE FROM fact_entities WHERE fact_id = ?", (fact_id,)
            )
            self._conn.execute("DELETE FROM facts WHERE fact_id = ?", (fact_id,))
            self._conn.commit()
            self._invalidate_vec_cache()
            self._rebuild_bank(row["category"])
            return True

    def list_facts(
        self,
        category: str | None = None,
        min_trust: float = 0.0,
        limit: int = 50,
    ) -> list[dict]:
        """Browse facts ordered by trust_score descending.

        Optionally filter by category and minimum trust score.
        """
        with self._lock:
            params: list = [min_trust]
            category_clause = ""
            if category is not None:
                category_clause = "AND category = ?"
                params.append(category)
            params.append(limit)

            sql = f"""
                SELECT fact_id, content, category, tags, trust_score,
                       retrieval_count, helpful_count, created_at, updated_at
                FROM facts
                WHERE trust_score >= ?
                  {category_clause}
                ORDER BY trust_score DESC
                LIMIT ?
            """
            rows = self._conn.execute(sql, params).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def record_feedback(self, fact_id: int, helpful: bool) -> dict:
        """Record user feedback and adjust trust asymmetrically.

        helpful=True  -> trust += 0.05, helpful_count += 1
        helpful=False -> trust -= 0.10

        Returns a dict with fact_id, old_trust, new_trust, helpful_count.
        Raises KeyError if fact_id does not exist.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT fact_id, trust_score, helpful_count FROM facts WHERE fact_id = ?",
                (fact_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"fact_id {fact_id} not found")

            old_trust: float = row["trust_score"]
            delta = _HELPFUL_DELTA if helpful else _UNHELPFUL_DELTA
            new_trust = _clamp_trust(old_trust + delta)

            helpful_increment = 1 if helpful else 0
            self._conn.execute(
                """
                UPDATE facts
                SET trust_score    = ?,
                    helpful_count  = helpful_count + ?,
                    updated_at     = CURRENT_TIMESTAMP
                WHERE fact_id = ?
                """,
                (new_trust, helpful_increment, fact_id),
            )
            self._conn.commit()

            return {
                "fact_id":      fact_id,
                "old_trust":    old_trust,
                "new_trust":    new_trust,
                "helpful_count": row["helpful_count"] + helpful_increment,
            }

    # ------------------------------------------------------------------
    # Entity helpers
    # ------------------------------------------------------------------

    def _extract_entities(self, text: str) -> list[str]:
        """Extract entity candidates from text using simple regex rules.

        Rules applied (in order):
        1. Capitalized multi-word phrases  e.g. "John Doe"
        2. Double-quoted terms             e.g. "Python"
        3. Single-quoted terms             e.g. 'pytest'
        4. AKA patterns                    e.g. "Guido aka BDFL" -> two entities
        5. CJK runs (2-8 chars)            e.g. "智谱茂莱光学" -> candidate spans
        6. Ticker patterns                 e.g. 688507 / 2513.HK / 002222.SZ

        Returns a deduplicated list preserving first-seen order.
        """
        seen: set[str] = set()
        candidates: list[str] = []

        def _add(name: str) -> None:
            stripped = name.strip()
            if stripped and stripped.lower() not in seen:
                seen.add(stripped.lower())
                candidates.append(stripped)

        # Distillation headers carry the date + subject entity: keep the
        # subject as a candidate, never the date (audit: 18.5% junk links
        # were date stamps eaten by the ticker regex).
        m = _RE_HEADER.match(text or "")
        if m and m.group(2):
            _add(m.group(2))
            text = _RE_HEADER.sub("", text, count=1)
        elif m:
            text = _RE_HEADER.sub("", text, count=1)

        for m in _RE_CAPITALIZED.finditer(text):
            _add(m.group(1))

        for m in _RE_DOUBLE_QUOTE.finditer(text):
            _add(m.group(1))

        for m in _RE_SINGLE_QUOTE.finditer(text):
            _add(m.group(1))

        for m in _RE_AKA.finditer(text):
            _add(m.group(1))
            _add(m.group(2))

        # CJK entity runs: a punctuation/latin-delimited run of 2-8 CJK
        # characters is a plausible entity span (company names, people,
        # domain terms). Whole-run capture keeps 智谱/茂莱光学 queryable;
        # prose sentences are filtered by the length cap.
        for m in _RE_CJK_RUN.finditer(text):
            run = m.group(0)
            if len(run) < 2:
                continue
            if len(run) <= 8:
                if run not in _ENTITY_STOPWORDS:
                    _add(run)
                continue
            # Over-long glued run (Chinese has no spaces): split at
            # function-word boundaries and keep 2-8 char sub-runs. This
            # recovers mid-sentence names (今天讨论智谱的估值 → 智谱)
            # that the whole-run cap used to swallow.
            for piece in _ENTITY_SPLIT_RE.split(run):
                piece = piece.strip()
                if 2 <= len(piece) <= 8 and piece not in _ENTITY_STOPWORDS:
                    _add(piece)

        for m in _RE_CAMEL_TOKEN.finditer(text):
            _add(m.group(0))

        # Ticker-style codes: bare 6-digit A-share codes (688507 / 002222 /
        # 301421 / 600895 / 000063 …), exchange-prefixed (sh600895), and
        # suffix forms (2513.HK / 002222.SZ). Date stamps must NOT match.
        for m in _RE_TICKER.finditer(text):
            _add(m.group(0))

        return candidates

    def _resolve_entity(self, name: str) -> int:
        """Find an existing entity by name or alias (case-insensitive) or create one.

        Returns the entity_id.
        """
        # Exact name match
        row = self._conn.execute(
            "SELECT entity_id FROM entities WHERE name LIKE ?", (name,)
        ).fetchone()
        if row is not None:
            return int(row["entity_id"])

        # Search aliases — aliases stored as comma-separated; use LIKE with % boundaries
        alias_row = self._conn.execute(
            """
            SELECT entity_id FROM entities
            WHERE ',' || aliases || ',' LIKE '%,' || ? || ',%'
            """,
            (name,),
        ).fetchone()
        if alias_row is not None:
            return int(alias_row["entity_id"])

        # Create new entity
        cur = self._conn.execute(
            "INSERT INTO entities (name) VALUES (?)", (name,)
        )
        self._conn.commit()
        return int(cur.lastrowid)  # type: ignore[return-value]

    def _link_fact_entity(self, fact_id: int, entity_id: int) -> None:
        """Insert into fact_entities, silently ignore if the link already exists."""
        self._conn.execute(
            """
            INSERT OR IGNORE INTO fact_entities (fact_id, entity_id)
            VALUES (?, ?)
            """,
            (fact_id, entity_id),
        )
        self._conn.commit()

    def _compute_hrr_vector(self, fact_id: int, content: str) -> None:
        """Compute and store HRR vector for a fact. No-op if numpy unavailable."""
        with self._lock:
            if not self._hrr_available:
                return

            # Get entities linked to this fact
            rows = self._conn.execute(
                """
                SELECT e.name FROM entities e
                JOIN fact_entities fe ON fe.entity_id = e.entity_id
                WHERE fe.fact_id = ?
                """,
                (fact_id,),
            ).fetchall()
            entities = [row["name"] for row in rows]

            vector = hrr.encode_fact(content, entities, self.hrr_dim)
            self._conn.execute(
                "UPDATE facts SET hrr_vector = ? WHERE fact_id = ?",
                (hrr.phases_to_bytes(vector), fact_id),
            )
            self._conn.commit()

    def _compute_embedding(self, fact_id: int, content: str) -> None:
        """Embed one fact for the semantic retrieval leg. No-op when the
        embedding client is missing or the call fails — lexical retrieval
        continues to work untouched (graceful degradation by design)."""
        if self._embed_client is None or not getattr(self._embed_client, "available", False):
            return
        try:
            from . import embedding as emb
        except ImportError:
            import embedding as emb  # type: ignore[no-redef]

        with self._lock:
            vec = self._embed_client.embed([content])
            if not vec:
                return
            self._conn.execute(
                "UPDATE facts SET embedding = ? WHERE fact_id = ?",
                (emb.pack_vector(vec[0]), fact_id),
            )
            self._conn.commit()

    def backfill_embeddings(self, batch_size: int = 16, max_batches: int | None = None) -> int:
        """Embed facts whose embedding column is NULL. Idempotent; returns
        the number of facts embedded this run. Failure of a batch just stops
        the run — already-written vectors stay, the next call resumes."""
        if self._embed_client is None or not getattr(self._embed_client, "available", False):
            return 0
        try:
            from . import embedding as emb
        except ImportError:
            import embedding as emb  # type: ignore[no-redef]

        done = 0
        with self._lock:
            batches = 0
            while max_batches is None or batches < max_batches:
                rows = self._conn.execute(
                    "SELECT fact_id, content FROM facts "
                    "WHERE embedding IS NULL AND COALESCE(content,'') != '' "
                    "ORDER BY fact_id LIMIT ?",
                    (batch_size,),
                ).fetchall()
                if not rows:
                    break
                vecs = self._embed_client.embed([r["content"] for r in rows])
                if not vecs:
                    break  # API down — resume next call
                for r, v in zip(rows, vecs):
                    self._conn.execute(
                        "UPDATE facts SET embedding = ? WHERE fact_id = ?",
                        (emb.pack_vector(v), r["fact_id"]),
                    )
                self._conn.commit()
                done += len(rows)
                batches += 1
        return done

    def _invalidate_vec_cache(self) -> None:
        """Fact mutation happened — next query rebuilds the matrix."""
        self._vec_matrix = None
        self._vec_ids = None

    def _build_vec_cache(self) -> bool:
        """Load all embeddings into one normalized numpy matrix. Returns True
        if the cache is usable (False → caller falls back to row-wise path)."""
        if not hrr._HAS_NUMPY or self._embed_client is None:
            return False
        import numpy as np
        with self._lock:
            n_total = self._conn.execute(
                "SELECT COUNT(*) FROM facts WHERE embedding IS NOT NULL"
            ).fetchone()[0]
            self._vec_stats["capped"] = n_total > self._vec_cache_max_facts
            if self._vec_stats["capped"]:
                return False
            rows = self._conn.execute(
                "SELECT fact_id, embedding FROM facts WHERE embedding IS NOT NULL"
            ).fetchall()
        if not rows:
            return False
        ids = [r["fact_id"] for r in rows]
        dim = len(rows[0]["embedding"]) // 4  # float32 blobs
        mat = np.empty((len(rows), dim), dtype=np.float32)
        for i, r in enumerate(rows):
            mat[i] = np.frombuffer(r["embedding"], dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._vec_matrix = mat / norms
        self._vec_ids = ids
        self._vec_stats["builds"] += 1
        return True

    def vector_candidates(
        self, query: str, top: int = 30, category: str | None = None
    ) -> list[dict]:
        """Semantic retrieval: embed the query once, cosine over stored fact
        embeddings. Returns [] when unavailable (no client, no vectors, or
        API failure) so callers fall back to lexical only.

        2026-08-30: serves from the in-RAM normalized matrix (one matmul per
        query) instead of re-fetching every blob per query. Falls back to the
        original row-wise path when numpy is missing or the store exceeds the
        cache capacity cap. The matrix is category-agnostic; category
        filtering is post-hoc (correct; curated tiers are small)."""
        if self._embed_client is None or not getattr(self._embed_client, "available", False):
            return []
        try:
            from . import embedding as emb
        except ImportError:
            import embedding as emb  # type: ignore[no-redef]

        if self._vec_matrix is None and not self._build_vec_cache():
            # Cache unavailable → original row-wise path, unchanged behavior
            with self._lock:
                where = "WHERE embedding IS NOT NULL"
                params: list = []
                if category:
                    where += " AND category = ?"
                    params.append(category)
                rows = self._conn.execute(
                    f"SELECT fact_id, content, category, tags, trust_score, "
                    f"retrieval_count, helpful_count, created_at, updated_at, embedding "
                    f"FROM facts {where}",
                    params,
                ).fetchall()
            if not rows:
                return []
            qv = self._embed_client.embed([query])
            if not qv:
                return []
            scored = []
            for r in rows:
                fact = dict(r)
                blob = fact.pop("embedding")
                scored.append((emb.cosine(qv[0], emb.unpack_vector(blob)), fact))
            scored.sort(key=lambda t: -t[0])
            return [dict(f, vsim=s) for s, f in scored[:top]]

        # Matrix path
        import numpy as np
        matrix = self._vec_matrix
        ids = self._vec_ids
        if matrix is None or ids is None:  # pragma: no cover - build just failed
            return []
        qv = self._embed_client.embed([query])
        if not qv:
            return []
        q = np.asarray(qv[0], dtype=np.float32)
        qn = np.linalg.norm(q)
        if qn == 0:
            return []
        q = q / qn
        sims = matrix @ q  # (n,)
        order = np.argsort(-sims)[: top * 4]  # overfetch for category filter
        if not order.size:
            return []
        wanted = [int(ids[i]) for i in order]
        ph = ",".join("?" * len(wanted))
        rows = self._conn.execute(
            f"SELECT fact_id, content, category, tags, trust_score, "
            f"retrieval_count, helpful_count, created_at, updated_at "
            f"FROM facts WHERE fact_id IN ({ph})", wanted).fetchall()
        by_id = {r["fact_id"]: dict(r) for r in rows}
        out = []
        for i in order:
            fid = int(ids[i])
            f = by_id.get(fid)
            if f is None:
                continue
            if category and f.get("category") != category:
                continue
            out.append(dict(f, vsim=float(sims[i])))
            if len(out) >= top:
                break
        self._vec_stats["hits"] += 1
        return out

    def _rebuild_bank(self, category: str) -> None:
        """Full rebuild of a category's memory bank from all its fact vectors."""
        with self._lock:
            if not self._hrr_available:
                return

            bank_name = f"cat:{category}"
            rows = self._conn.execute(
                "SELECT hrr_vector FROM facts WHERE category = ? AND hrr_vector IS NOT NULL",
                (category,),
            ).fetchall()

            if not rows:
                self._conn.execute("DELETE FROM memory_banks WHERE bank_name = ?", (bank_name,))
                self._conn.commit()
                return

            vectors = [hrr.bytes_to_phases(row["hrr_vector"], dim=self.hrr_dim) for row in rows]
            bank_vector = hrr.bundle(*vectors)
            fact_count = len(vectors)

            # Check SNR
            hrr.snr_estimate(self.hrr_dim, fact_count)

            self._conn.execute(
                """
                INSERT INTO memory_banks (bank_name, vector, dim, fact_count, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(bank_name) DO UPDATE SET
                    vector = excluded.vector,
                    dim = excluded.dim,
                    fact_count = excluded.fact_count,
                    updated_at = excluded.updated_at
                """,
                (bank_name, hrr.phases_to_bytes(bank_vector), self.hrr_dim, fact_count),
            )
            self._conn.commit()

    def rebuild_all_vectors(self, dim: int | None = None) -> int:
        """Recompute all HRR vectors + banks from text. For recovery/migration.

        Returns the number of facts processed.
        """
        with self._lock:
            if not self._hrr_available:
                return 0

            if dim is not None:
                self.hrr_dim = dim

            rows = self._conn.execute(
                "SELECT fact_id, content, category FROM facts"
            ).fetchall()

            categories: set[str] = set()
            for row in rows:
                self._compute_hrr_vector(row["fact_id"], row["content"])
                categories.add(row["category"])

            for category in categories:
                self._rebuild_bank(category)

            return len(rows)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        """Convert a sqlite3.Row to a plain dict."""
        return dict(row)

    @classmethod
    def release_all_under(cls, directory: "str | Path") -> int:
        """Force-close every shared connection whose database lives under ``directory``.

        ``close()`` is refcount-driven, so a live holder (e.g. an agent's
        memory provider) keeps a profile's SQLite handle open indefinitely.
        That is exactly what a profile delete must break on Windows: the
        desktop's main ``serve`` process opens ``memory_store.db`` for every
        known profile, and ``rmtree`` of the profile directory fails with
        ``WinError 32`` while any of those handles is open (#88347). This
        closes the matching connections unconditionally — the directory is
        going away, so later use by a stale holder is expected to fail — and
        returns how many were closed. In a process that holds none (e.g. the
        CLI deleting from outside serve) this is a harmless no-op returning 0.
        """
        root = os.path.normcase(str(Path(directory).expanduser().resolve())) + os.sep
        with cls._shared_guard:
            # Snapshot the keys first so the registry stays stable while
            # connections are closed inside their per-database locks (closing
            # can run no user code, but this keeps the invariant obvious).
            doomed = [
                key
                for key in cls._shared
                if os.path.normcase(key).startswith(root)
            ]
            for key in doomed:
                entry = cls._shared.pop(key)
                try:
                    with entry["lock"]:
                        entry["conn"].close()
                except Exception:
                    # A connection that is already closed or broken must not
                    # abort releasing its siblings.
                    pass
        return len(doomed)

    def close(self) -> None:
        """Release this instance's reference to the shared connection.

        The underlying connection is closed only when the last MemoryStore
        referencing the same database is closed, so closing one instance can
        never break sibling instances that still hold it. Idempotent.
        """
        if getattr(self, "_entry", None) is None:
            return
        with MemoryStore._shared_guard:
            entry = self._entry
            if entry is None:
                return
            entry["refs"] -= 1
            if entry["refs"] <= 0:
                try:
                    entry["conn"].close()
                finally:
                    # Pop only OUR entry. After release_all_under() force-
                    # closed this entry (profile delete, #88347) a same-path
                    # store may have re-registered a FRESH entry under the
                    # same key; a stale holder's late close() must not evict
                    # it — that would silently reintroduce the multi-writer
                    # contention this registry exists to prevent.
                    if MemoryStore._shared.get(self._key) is entry:
                        MemoryStore._shared.pop(self._key, None)
            self._entry = None

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
