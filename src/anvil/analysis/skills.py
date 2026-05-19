"""Audit skills across all three AI tools: what you have installed, what gets exposed in
Cursor, and what actually gets used anywhere.

Cursor injects an ``<available_skills>`` block into every prompt that lists ``<agent_skill
fullPath="...">description</agent_skill>`` entries. The full body of each skill is only
loaded when the agent decides to ``Read`` the path. That means:

* Every installed skill costs ~its-description-length tokens in *every* Cursor session.
* Skills whose descriptions don't match your actual asks are pure overhead - paid for
  but never opened.

Detection rules
---------------

We match skills by **name** (the directory containing SKILL.md) rather than full path, so
the same skill installed under both ``~/.cursor/skills/`` and ``~/.codex/skills/`` counts
as one. Consultation evidence comes from any of three sources:

* **Cursor**: an assistant-role turn mentions ``/<name>/SKILL.md``. (Cursor transcripts
  don't expose structured tool calls, so this is the cleanest text-only signal.)
* **Codex**: a ``response_item`` event with ``payload.type == "message"`` and ``role`` in
  ``{user, assistant}`` mentions the skill name. The catalog injection rides on
  ``role == "developer"`` and is filtered out structurally, not by string matching.
* **Claude Code**: a ``type == "assistant"`` or ``type == "user"`` event mentions the
  skill name. Other types (``file-history-snapshot``, ``last-prompt``) are dropped.

A single event listing ≥4 distinct skill names is treated as a catalog echo, not a
consultation. Real conversational invocations name one to a few skills in prose; bulk
listings come from the agent reading or echoing the catalog itself.

Skill names are looked up as either ``/<name>/SKILL.md`` (path form) or a literal
hyphen-bearing token (e.g. ``baz-pr-comments``), so a turn that says "use the
baz-pr-comments skill" without quoting a path still counts.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from anvil.parsers.cursor_transcripts import CursorScanResult, estimate_tokens

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
# Matches any '.../<skill-name>/SKILL.md' and captures the directory name as the skill's
# canonical key, so identical skills installed in multiple roots collapse to one entry.
_SKILL_NAME_RE = re.compile(r"/([a-zA-Z0-9._-]+)/SKILL\.md")
# A bare kebab-case identifier that could be a skill name in prose ("use baz-pr-comments").
# We require at least one hyphen to keep precision; single-word skill names (e.g. `imagegen`)
# only count via path-form mentions.
_KEBAB_TOKEN_RE = re.compile(r"\b([a-z][a-z0-9]*(?:-[a-z0-9]+)+)\b")
# Above this many distinct skill names per single event, we assume the event is a catalog
# echo (assistant listing every skill, agent ls-ing the skills dir) rather than a real
# consultation. Genuine invocations name 1-3 skills in prose.
_CATALOG_ECHO_THRESHOLD = 3

SkillSource = Literal["codex", "claude"]


@dataclass
class SkillRecord:
    """One on-disk skill (a directory with a SKILL.md inside)."""

    name: str
    full_path: Path
    description: str | None
    body_bytes: int
    body_tokens: int
    description_tokens: int
    last_modified_ts: float

    # Populated by cross_reference():
    exposed_in_sessions: int = 0  # Cursor sessions that listed it in <available_skills>
    consulted_in_sessions: int = 0  # consultation evidence across Cursor + Codex + Claude Code

    @property
    def is_stale(self) -> bool:
        """Exposed in ≥10 Cursor sessions with zero visible consultation across all three sources.

        May still be load-bearing - the model could be following it silently in Cursor without
        quoting the SKILL.md path. "Stale" means "no visible evidence of consultation in any
        transcript we can see", not "definitively unused".
        """
        return self.exposed_in_sessions >= 10 and self.consulted_in_sessions == 0

    @property
    def utilization(self) -> float:
        """consulted / exposed, clamped 0..1. None-equivalent (0.0) when never exposed."""
        if self.exposed_in_sessions == 0:
            return 0.0
        return min(1.0, self.consulted_in_sessions / self.exposed_in_sessions)


@dataclass
class SkillsAuditReport:
    """Top-level result: per-skill stats plus rolled-up totals."""

    skills: list[SkillRecord]
    total_sessions_scanned: int
    # Per-session overhead = sum of every skill description that gets injected each session.
    # We approximate it as sum(description_tokens) since Cursor lists every installed skill.
    per_session_description_overhead_tokens: int = 0
    total_lifetime_description_tokens: int = 0  # overhead times sessions
    stale_skills_count: int = 0  # skills exposed ≥10 sessions with no visible consultation

    @property
    def total_skills(self) -> int:
        return len(self.skills)

    @property
    def total_body_tokens(self) -> int:
        return sum(s.body_tokens for s in self.skills)


def _parse_frontmatter(text: str) -> dict[str, str]:
    """Pull simple ``key: value`` pairs out of YAML frontmatter. We don't need full YAML."""
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        return {}
    out: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def scan_skills(skills_root: Path) -> list[SkillRecord]:
    """Find every SKILL.md under ``skills_root`` (one level deep) and parse it."""
    if not skills_root.exists() or not skills_root.is_dir():
        return []
    records: list[SkillRecord] = []
    # We look for both <skill_dir>/SKILL.md and any depth-2 SKILL.md to support nested layouts
    # like ~/.codex/skills/.system/skill-creator/SKILL.md.
    for skill_md in sorted(skills_root.rglob("SKILL.md")):
        try:
            text = skill_md.read_text(errors="replace")
        except OSError:
            continue
        frontmatter = _parse_frontmatter(text)
        name = frontmatter.get("name") or skill_md.parent.name
        description = frontmatter.get("description")
        records.append(
            SkillRecord(
                name=name,
                full_path=skill_md,
                description=description,
                body_bytes=len(text.encode("utf-8")),
                body_tokens=estimate_tokens(text),
                description_tokens=estimate_tokens(description or ""),
                last_modified_ts=skill_md.stat().st_mtime,
            )
        )
    return records


def _collect_session_skill_signals(scan: CursorScanResult) -> tuple[dict[str, int], dict[str, int]]:
    """Walk Cursor transcripts once, return (exposed_count_by_name, consulted_count_by_name).

    "Exposed" = a user-side turn lists ``/<name>/SKILL.md`` (Cursor's injected catalog).
    "Consulted" = an assistant-side turn mentions ``/<name>/SKILL.md`` (model quoted the path).
    Keyed on the skill's directory name so cross-source matching works downstream.
    """
    exposed: dict[str, int] = defaultdict(int)
    consulted: dict[str, int] = defaultdict(int)
    for session in scan.sessions:
        exposed_in_this_session: set[str] = set()
        consulted_in_this_session: set[str] = set()
        for turn in session.turns:
            names = {m.group(1) for m in _SKILL_NAME_RE.finditer(turn.raw_text)}
            if turn.role == "user":
                exposed_in_this_session |= names
            else:
                consulted_in_this_session |= names
        for n in exposed_in_this_session:
            exposed[n] += 1
        for n in consulted_in_this_session:
            consulted[n] += 1
    return dict(exposed), dict(consulted)


def _extract_conversational_text(event: dict[str, object], source: SkillSource) -> str | None:
    """Return the text body of a real user/assistant conversation event, or None to skip.

    This is where heuristics turn into structural rules: each source has its own event
    taxonomy, and the catalog injections / tool output dumps live in known event types we
    can drop outright.
    """
    if source == "codex":
        if event.get("type") != "response_item":
            return None
        payload = event.get("payload") or {}
        if not isinstance(payload, dict) or payload.get("type") != "message":
            return None
        # role=="developer" is where Codex injects the skill catalog and tool docs.
        # role=="system" is similarly system-side. Only user/assistant turns count.
        if payload.get("role") not in {"user", "assistant"}:
            return None
        content = payload.get("content")
        if not isinstance(content, list):
            return None
        return "".join(c.get("text", "") for c in content if isinstance(c, dict) and isinstance(c.get("text"), str))
    # claude
    if event.get("type") not in {"user", "assistant"}:
        return None
    message = event.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for c in content:
            if isinstance(c, dict):
                # text blocks, tool_use blocks (input may contain a SKILL.md path), tool_result blocks
                if isinstance(c.get("text"), str):
                    parts.append(c["text"])
                if isinstance(c.get("input"), dict):
                    parts.append(json.dumps(c["input"]))
                if isinstance(c.get("content"), str):
                    parts.append(c["content"])
        return "".join(parts)
    return None


def _skill_names_in_text(text: str, known_names: set[str]) -> set[str]:
    """Pull skill-name mentions out of one event's text body.

    Matches both ``/<name>/SKILL.md`` paths and bare kebab-case tokens like
    ``baz-pr-comments``, then intersects with the set of skills the user actually has
    installed so random kebab-case identifiers (commit messages, file names) don't
    register.
    """
    path_names = {m.group(1) for m in _SKILL_NAME_RE.finditer(text)}
    kebab_names = {m.group(1) for m in _KEBAB_TOKEN_RE.finditer(text)}
    return (path_names | kebab_names) & known_names


def scan_files_for_skill_mentions(
    files: Iterable[Path],
    *,
    source: SkillSource,
    known_names: set[str],
) -> dict[str, int]:
    """Scan JSONL transcripts by parsing structured events. Returns {name: session_count}.

    Per session we keep only user/assistant message events (Codex ``response_item.message``
    with ``role in {user, assistant}``, or Claude Code ``type in {user, assistant}``) and
    drop everything else - so catalog injections (Codex ``role=developer``), tool output
    dumps (``exec_command_end``, ``function_call_output``), and file-history snapshots
    never feed the consultation signal in the first place.

    Within retained events we still apply one bulk-listing filter: any single event
    naming ≥4 distinct skill names is treated as the assistant echoing the catalog or
    listing many skills at once, not consulting them. Genuine conversational references
    name one to a few skills.
    """
    counts: dict[str, int] = defaultdict(int)
    for path in files:
        consulted_in_this_session: set[str] = set()
        try:
            with path.open(encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    try:
                        event = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if not isinstance(event, dict):
                        continue
                    text = _extract_conversational_text(event, source)
                    if text is None or not text:
                        continue
                    names = _skill_names_in_text(text, known_names)
                    if 0 < len(names) <= _CATALOG_ECHO_THRESHOLD:
                        consulted_in_this_session |= names
        except OSError:
            continue
        for name in consulted_in_this_session:
            counts[name] += 1
    return dict(counts)


def build_skills_audit(
    skills: list[SkillRecord],
    scan: CursorScanResult,
    extra_consulted_by_name: dict[str, int] | None = None,
) -> SkillsAuditReport:
    """Cross-reference installed skills with transcript signals and roll up totals.

    ``extra_consulted_by_name`` lets callers add session counts from non-Cursor sources
    (Codex, Claude Code) keyed by skill directory name.
    """
    exposed, consulted = _collect_session_skill_signals(scan)
    extra = extra_consulted_by_name or {}
    enriched: list[SkillRecord] = []
    for skill in skills:
        # The skill's directory name is the canonical key. For skills nested under
        # ``.system/`` or similar parents we still use the immediate parent directory.
        key = skill.full_path.parent.name
        enriched.append(
            SkillRecord(
                name=skill.name,
                full_path=skill.full_path,
                description=skill.description,
                body_bytes=skill.body_bytes,
                body_tokens=skill.body_tokens,
                description_tokens=skill.description_tokens,
                last_modified_ts=skill.last_modified_ts,
                exposed_in_sessions=exposed.get(key, 0),
                consulted_in_sessions=consulted.get(key, 0) + extra.get(key, 0),
            )
        )
    # Sort: stale skills first (worst offenders), then by descending body size.
    enriched.sort(key=lambda s: (not s.is_stale, -s.body_tokens))
    total_sessions = len(scan.sessions)
    overhead = sum(s.description_tokens for s in enriched)
    return SkillsAuditReport(
        skills=enriched,
        total_sessions_scanned=total_sessions,
        per_session_description_overhead_tokens=overhead,
        total_lifetime_description_tokens=overhead * total_sessions,
        stale_skills_count=sum(1 for s in enriched if s.is_stale),
    )
