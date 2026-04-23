#!/usr/bin/env bash
# PreCompact hook: Inject state preservation guidance before context compaction.

cd "$CLAUDE_PROJECT_DIR" 2>/dev/null || exit 0

STATE=""

BRANCH=$(git branch --show-current 2>/dev/null)
[ -n "$BRANCH" ] && STATE="${STATE}Branch: ${BRANCH}\n"

DIRTY=$(git status --porcelain 2>/dev/null)
if [ -n "$DIRTY" ]; then
    COUNT=$(echo "$DIRTY" | wc -l | tr -d ' ')
    STATE="${STATE}Uncommitted files (${COUNT}):\n${DIRTY}\n"
fi

UPSTREAM=$(git rev-parse --abbrev-ref '@{upstream}' 2>/dev/null)
if [ -n "$UPSTREAM" ]; then
    AHEAD=$(git rev-list --count "${UPSTREAM}..HEAD" 2>/dev/null)
    [ "$AHEAD" -gt 0 ] 2>/dev/null && STATE="${STATE}Unpushed commits: ${AHEAD}\n"
fi

RECENT=$(git log --oneline -5 2>/dev/null)
[ -n "$RECENT" ] && STATE="${STATE}Recent commits:\n${RECENT}\n"

LATEST_HANDOFF=$(ls -t "$CLAUDE_PROJECT_DIR/.claude/handoffs/"*.md 2>/dev/null | head -1)
if [ -n "$LATEST_HANDOFF" ] && [ -f "$LATEST_HANDOFF" ]; then
    HANDOFF_CONTENT=$(head -40 "$LATEST_HANDOFF" 2>/dev/null)
    [ -n "$HANDOFF_CONTENT" ] && STATE="${STATE}\nHandoff context:\n${HANDOFF_CONTENT}\n"
fi

STATE="${STATE}\nProject conventions to preserve:\n"
STATE="${STATE}- Python 3.9+, uv for all tooling, ruff + mypy via prek\n"
STATE="${STATE}- Verification: uv run prek (single command for lint/format/types)\n"
STATE="${STATE}- Pre-push: uv run prek run --from-ref origin/<base>\n"
STATE="${STATE}- Conventional commits: fix:, feat:, refactor:, test:, chore:\n"
STATE="${STATE}- Result type: Success(value) / Failure(error), check with is_successful()\n"
STATE="${STATE}- Language singleton: set_current_language() / current_language()\n"
STATE="${STATE}- libcst for code transforms, ast for read-only analysis\n"

[ -z "$STATE" ] && exit 0

EXPANDED=$(printf '%b' "$STATE")
jq -n --arg msg "PRESERVE the following session state through compaction:
$EXPANDED" '{"systemMessage": $msg}'

exit 0
