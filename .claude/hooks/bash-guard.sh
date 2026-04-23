#!/usr/bin/env bash
# PreToolUse hook: Block Bash calls that should use dedicated tools.
# Exit 0 = allow, Exit 2 = block (message on stderr).

INPUT=$(cat 2>/dev/null || true)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty' 2>/dev/null || true)

[ -z "$COMMAND" ] && exit 0

# Strip leading env vars (FOO=bar cmd ...) and whitespace to get the actual command
STRIPPED=$(echo "$COMMAND" | sed 's/^[[:space:]]*\([A-Za-z_][A-Za-z0-9_]*=[^[:space:]]*[[:space:]]*\)*//')
FIRST_CMD=$(echo "$STRIPPED" | awk '{print $1}')

case "$FIRST_CMD" in
    grep|egrep|fgrep|rg)
        echo "BLOCKED: Use the Grep tool instead of \`$FIRST_CMD\`. It provides better output and permissions handling." >&2
        exit 2
        ;;
    find)
        echo "BLOCKED: Use the Glob tool instead of \`find\`. Glob is faster and returns results sorted by modification time." >&2
        exit 2
        ;;
    cat|head|tail)
        echo "BLOCKED: Use the Read tool instead of \`$FIRST_CMD\`. Read provides line numbers and supports images/PDFs." >&2
        exit 2
        ;;
    sed)
        if echo "$COMMAND" | grep -qE '(^|[[:space:]])sed[[:space:]]+-i'; then
            echo "BLOCKED: Use the Edit tool instead of \`sed -i\`. Edit tracks changes properly." >&2
            exit 2
        fi
        ;;
esac

# echo with file redirection (echo "..." > file)
if echo "$STRIPPED" | grep -qE '^echo\b.*[[:space:]]>'; then
    echo "BLOCKED: Use the Write tool instead of \`echo >\`. Write provides proper file creation." >&2
    exit 2
fi

exit 0
