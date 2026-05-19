"""GitHub connector for fetching the user's PR + commit activity.

Uses the ``gh`` CLI (already authed on the user's machine) instead of raw HTTP, so we
get free token management, refresh, and 2FA handling. If ``gh`` is not installed or
not authed, we surface a clear error.

The MVP focuses on **PRs authored by the user** since that's the cleanest shipping
signal. We capture: number, title, repo, state, merged-at, created-at, additions,
deletions, changed_files. Commit-level data is intentionally skipped for v1 -
PR-level signal is far less noisy.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


class GitHubConnectorError(RuntimeError):
    pass


@dataclass
class PullRequestRecord:
    number: int
    repo: str  # "owner/name"
    title: str
    state: str  # OPEN, CLOSED, MERGED
    created_at: datetime
    merged_at: datetime | None
    closed_at: datetime | None
    additions: int
    deletions: int
    changed_files: int
    is_draft: bool
    url: str
    body: str = ""  # PR description; used by the significance classifier for keyword signals
    review_count: int = 0  # how many distinct reviewers approved/commented

    @property
    def shipped(self) -> bool:
        """True if the PR was merged. Closed-without-merge counts as did-not-ship."""
        return self.merged_at is not None

    @property
    def total_lines_changed(self) -> int:
        return self.additions + self.deletions


def _check_gh_available() -> None:
    if shutil.which("gh") is None:
        raise GitHubConnectorError(
            "gh CLI not found in PATH. Install: https://cli.github.com/ - then run `gh auth login`."
        )


def _check_gh_authed() -> str:
    """Return the authenticated login. Raises if not authed."""
    try:
        result = subprocess.run(
            ["gh", "api", "user", "--jq", ".login"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise GitHubConnectorError(f"gh not authed - run `gh auth login`. stderr: {exc.stderr}") from exc
    except subprocess.TimeoutExpired as exc:
        raise GitHubConnectorError("gh api user timed out") from exc
    return result.stdout.strip()


def current_login() -> str:
    """Return the authenticated GitHub login (raises if unavailable)."""
    _check_gh_available()
    return _check_gh_authed()


_PR_SEARCH_QUERY = """
query($search: String!, $cursor: String) {
  search(query: $search, type: ISSUE, first: 100, after: $cursor) {
    pageInfo { hasNextPage endCursor }
    nodes {
      __typename
      ... on PullRequest {
        number
        title
        body
        state
        isDraft
        url
        createdAt
        mergedAt
        closedAt
        additions
        deletions
        changedFiles
        repository { nameWithOwner }
        reviews(first: 1) { totalCount }
      }
    }
  }
}
"""


def _parse_pr(node: dict) -> PullRequestRecord:  # type: ignore[type-arg]
    return PullRequestRecord(
        number=int(node["number"]),
        repo=str(node["repository"]["nameWithOwner"]),
        title=str(node["title"]),
        state=str(node["state"]),
        created_at=datetime.fromisoformat(node["createdAt"].replace("Z", "+00:00")),
        merged_at=(datetime.fromisoformat(node["mergedAt"].replace("Z", "+00:00")) if node.get("mergedAt") else None),
        closed_at=(datetime.fromisoformat(node["closedAt"].replace("Z", "+00:00")) if node.get("closedAt") else None),
        additions=int(node.get("additions") or 0),
        deletions=int(node.get("deletions") or 0),
        changed_files=int(node.get("changedFiles") or 0),
        is_draft=bool(node.get("isDraft", False)),
        url=str(node["url"]),
        body=str(node.get("body") or ""),
        review_count=int((node.get("reviews") or {}).get("totalCount") or 0),
    )


def fetch_authored_prs(
    login: str | None = None,
    *,
    since: datetime | None = None,
    timeout_seconds: int = 30,
) -> list[PullRequestRecord]:
    """Fetch PRs authored by ``login`` (or the current user) since ``since``.

    Paginates through the GitHub Search API via ``gh api graphql``. Caps at 1000 results.
    """
    _check_gh_available()
    if login is None:
        login = _check_gh_authed()

    # Build the search query. Restrict to PRs authored by this user.
    parts = [f"author:{login}", "type:pr"]
    if since is not None:
        # YYYY-MM-DD is what GitHub search accepts.
        parts.append(f"created:>={since.date().isoformat()}")
    search = " ".join(parts)

    prs: list[PullRequestRecord] = []
    cursor: str | None = None
    pages = 0

    while True:
        args = [
            "gh",
            "api",
            "graphql",
            "-f",
            f"search={search}",
            "-f",
            f"query={_PR_SEARCH_QUERY}",
        ]
        if cursor is not None:
            args.extend(["-f", f"cursor={cursor}"])

        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise GitHubConnectorError(f"gh graphql failed: {exc.stderr}") from exc
        except subprocess.TimeoutExpired as exc:
            raise GitHubConnectorError(f"gh graphql timed out after {timeout_seconds}s") from exc

        payload = json.loads(result.stdout)
        if "errors" in payload:
            raise GitHubConnectorError(f"GraphQL errors: {payload['errors']}")
        search_node = payload["data"]["search"]
        for node in search_node["nodes"]:
            if node.get("__typename") == "PullRequest":
                prs.append(_parse_pr(node))

        page_info = search_node["pageInfo"]
        pages += 1
        if not page_info["hasNextPage"] or pages >= 10:
            break
        cursor = page_info["endCursor"]

    return prs


def default_since(days: int = 90) -> datetime:
    """Sensible default lookback for anvil analyses."""
    return datetime.now(tz=UTC) - timedelta(days=days)
