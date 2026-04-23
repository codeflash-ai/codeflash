#!/usr/bin/env bash
# PreToolUse hook: Block Write/Edit on existing files that haven't been Read first.
# Exit 0 = allow, Exit 2 = block (message on stderr).

INPUT=$(cat 2>/dev/null || true)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty' 2>/dev/null || true)

[ -z "$FILE_PATH" ] && exit 0

# New files don't need prior reads
[ ! -f "$FILE_PATH" ] && exit 0

TRACKER="$CLAUDE_PROJECT_DIR/.claude/.read-tracker"

if [ ! -f "$TRACKER" ]; then
    echo "BLOCKED: Read \`$(basename "$FILE_PATH")\` first before modifying it." >&2
    exit 2
fi

if grep -qxF "$FILE_PATH" "$TRACKER"; then
    exit 0
fi

echo "BLOCKED: Read \`$(basename "$FILE_PATH")\` first before modifying it." >&2
exit 2
