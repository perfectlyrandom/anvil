"""In-process cache for scan results, keyed on the tree's max mtime.

The dashboard re-scans the same projects directory for every tab switch, which
walks 500+ JSONL files and takes seconds. This cache holds the parsed
:class:`CursorScanResult` / :class:`ClaudeCodeScanResult` in memory and only
re-parses when something actually changed on disk.

Invalidation strategy: cache key includes the max ``st_mtime`` across the tree
*and* the file count. If you add/remove a transcript or the latest session
file is appended to, the key changes and we re-scan. Cheap (`Path.stat()` only,
no JSON parsing) and correct enough for a single-user local dashboard.

A 60-second floor TTL protects against extremely fast tab switches that could
otherwise back up on a slow mtime walk; we only check the tree once per minute
unless a re-scan was explicitly requested.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

from anvil.analysis.skills import SkillRecord, SkillSource, scan_files_for_skill_mentions, scan_skills
from anvil.connectors.github import (
    GitHubConnectorError,
    PullRequestRecord,
    default_since,
    fetch_authored_prs,
    fetch_reviews_given_count,
)
from anvil.parsers.claude_code import ClaudeCodeScanResult, scan_claude_code_projects
from anvil.parsers.codex import CodexScanResult, scan_codex_sessions
from anvil.parsers.cursor_bubbles import CursorBubbleResult, scan_cursor_bubbles
from anvil.parsers.cursor_tracking import CursorTrackingResult, scan_cursor_tracking
from anvil.parsers.cursor_transcripts import CursorScanResult, scan_cursor_projects

logger = logging.getLogger(__name__)

T = TypeVar("T")

# How long we trust the cached mtime/file-count probe before re-checking the tree.
_PROBE_TTL_SECONDS = 60.0
# GitHub doesn't have a cheap "did anything change?" probe; trust a flat TTL.
_GITHUB_TTL_SECONDS = 300.0


@dataclass
class _CacheEntry(Generic[T]):
    value: T
    cache_key: tuple[float, int]
    probed_at: float


def _scan_signature(directory: Path, glob_pattern: str) -> tuple[float, int]:
    """Cheap fingerprint of the tree: (max mtime, file count). No JSON parsing."""
    if not directory.exists():
        return (0.0, 0)
    paths = list(directory.glob(glob_pattern))
    if not paths:
        return (0.0, 0)
    max_mtime = max(p.stat().st_mtime for p in paths)
    return (max_mtime, len(paths))


class ScanCache:
    """Thread-safe mtime-keyed cache for project scans.

    One instance lives on the FastAPI app and is shared across requests.
    Each fragment route asks for the latest scan; the cache hands back the
    cached result if the tree hasn't moved, or re-scans if it has.
    """

    def __init__(self) -> None:
        self._cursor: _CacheEntry[CursorScanResult] | None = None
        self._claude: _CacheEntry[ClaudeCodeScanResult] | None = None
        self._codex: _CacheEntry[CodexScanResult] | None = None
        self._cursor_tracking: tuple[float, float, CursorTrackingResult] | None = None  # (mtime, probed_at, value)
        self._cursor_bubbles: tuple[float, float, CursorBubbleResult] | None = None  # (mtime, probed_at, value)
        self._skills: _CacheEntry[list[SkillRecord]] | None = None
        # Skill-name consultation counts from non-Cursor sources, keyed by source label.
        self._skill_mentions: dict[str, _CacheEntry[dict[str, int]]] = {}
        self._prs: dict[tuple[str, int], tuple[float, list[PullRequestRecord]]] = {}
        self._reviews_given: dict[tuple[str, int], tuple[float, int | None]] = {}
        self._lock = threading.Lock()

    def _get(
        self,
        entry: _CacheEntry[T] | None,
        directory: Path,
        glob_pattern: str,
        scan_fn: Callable[[Path], T],
        label: str,
    ) -> tuple[_CacheEntry[T], T]:
        now = time.monotonic()
        # Probe rate-limit: don't re-stat the whole tree on every keystroke.
        if entry is not None and now - entry.probed_at < _PROBE_TTL_SECONDS:
            return entry, entry.value
        signature = _scan_signature(directory, glob_pattern)
        if entry is not None and entry.cache_key == signature:
            entry.probed_at = now
            return entry, entry.value
        start = time.monotonic()
        value = scan_fn(directory)
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "%s scan: %d files, %.0fms (cache miss, mtime=%s)",
            label,
            signature[1],
            elapsed_ms,
            signature[0],
        )
        new_entry = _CacheEntry(value=value, cache_key=signature, probed_at=now)
        return new_entry, value

    def cursor(self, directory: Path) -> CursorScanResult:
        with self._lock:
            self._cursor, value = self._get(
                self._cursor,
                directory,
                "*/agent-transcripts/**/*.jsonl",
                scan_cursor_projects,
                "cursor",
            )
            return value

    def claude(self, directory: Path) -> ClaudeCodeScanResult:
        with self._lock:
            self._claude, value = self._get(
                self._claude,
                directory,
                "**/*.jsonl",
                scan_claude_code_projects,
                "claude",
            )
            return value

    def skills(self, directory: Path) -> list[SkillRecord]:
        with self._lock:
            self._skills, value = self._get(
                self._skills,
                directory,
                "**/SKILL.md",
                scan_skills,
                "skills",
            )
            return value

    def codex(self, directory: Path) -> CodexScanResult:
        with self._lock:
            self._codex, value = self._get(
                self._codex,
                directory,
                "**/rollout-*.jsonl",
                scan_codex_sessions,
                "codex",
            )
            return value

    def skill_mentions(
        self,
        directory: Path,
        glob_pattern: str,
        source: SkillSource,
        known_names: frozenset[str],
    ) -> dict[str, int]:
        """Scan JSONL transcripts under ``directory`` for skill-name mentions.

        ``source`` selects the per-tool event filter ("codex" vs "claude"). ``known_names``
        is the set of skill directory names we'll match against bare kebab-case tokens in
        prose. Returns ``{name: session_count}`` - how many sessions mention each skill.
        Cache is keyed on (source, known_names) so adding/removing a skill invalidates.
        """
        with self._lock:
            cache_key = f"{source}:{hash(known_names)}"
            entry = self._skill_mentions.get(cache_key)

            def _scan(d: Path) -> dict[str, int]:
                return scan_files_for_skill_mentions(
                    d.glob(glob_pattern) if d.exists() else [],
                    source=source,
                    known_names=set(known_names),
                )

            new_entry, value = self._get(entry, directory, glob_pattern, _scan, f"skill-mentions/{source}")
            self._skill_mentions[cache_key] = new_entry
            return value

    def cursor_tracking(self, db_path: Path) -> CursorTrackingResult:
        """Cache the Cursor AI-tracking DB read with a simple mtime key."""
        with self._lock:
            now = time.monotonic()
            mtime = db_path.stat().st_mtime if db_path.exists() else 0.0
            cached = self._cursor_tracking
            if cached is not None and cached[0] == mtime and now - cached[1] < _PROBE_TTL_SECONDS:
                return cached[2]
            value = scan_cursor_tracking(db_path)
            self._cursor_tracking = (mtime, now, value)
            return value

    def cursor_bubbles(self, db_path: Path) -> CursorBubbleResult:
        """Cache the Cursor state.vscdb bubble token scan with a simple mtime key.

        This DB is big (600K+ rows) so we don't want to re-read it on every fragment
        request. mtime-keyed cache + a probe-TTL handles both 'file untouched' and
        'we just looked, give it a sec' cases.
        """
        with self._lock:
            now = time.monotonic()
            mtime = db_path.stat().st_mtime if db_path.exists() else 0.0
            cached = self._cursor_bubbles
            if cached is not None and cached[0] == mtime and now - cached[1] < _PROBE_TTL_SECONDS:
                return cached[2]
            value = scan_cursor_bubbles(db_path)
            self._cursor_bubbles = (mtime, now, value)
            return value

    def prs(self, login: str, days: int) -> list[PullRequestRecord]:
        """Cached PR fetch keyed on (login, days). Flat 5-minute TTL."""
        key = (login, days)
        now = time.monotonic()
        with self._lock:
            entry = self._prs.get(key)
            if entry is not None and now - entry[0] < _GITHUB_TTL_SECONDS:
                return entry[1]
        # Do the (slow) network call OUTSIDE the lock so a slow gh doesn't block other tabs.
        start = time.monotonic()
        prs = fetch_authored_prs(login, since=default_since(days))
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info("github prs fetch: %d prs, %.0fms (login=%s, days=%d)", len(prs), elapsed_ms, login, days)
        with self._lock:
            self._prs[key] = (time.monotonic(), prs)
        return prs

    def reviews_given_count(self, login: str, days: int) -> int | None:
        """Cached count of PRs the user reviewed in the window. None on fetch error."""
        key = (login, days)
        now = time.monotonic()
        with self._lock:
            entry = self._reviews_given.get(key)
            if entry is not None and now - entry[0] < _GITHUB_TTL_SECONDS:
                return entry[1]
        try:
            count: int | None = fetch_reviews_given_count(login, since=default_since(days))
        except GitHubConnectorError as exc:
            logger.warning("reviews-given fetch failed (login=%s): %s", login, exc)
            count = None
        with self._lock:
            self._reviews_given[key] = (time.monotonic(), count)
        return count

    def invalidate(self) -> None:
        """Force a re-scan on the next access. For debugging / manual refresh."""
        with self._lock:
            self._cursor = None
            self._claude = None
            self._codex = None
            self._cursor_tracking = None
            self._cursor_bubbles = None
            self._skills = None
            self._skill_mentions.clear()
            self._prs.clear()
