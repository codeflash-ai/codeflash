#!/usr/bin/env bash
# PostToolUse hook: Track Read calls for the require-read guard.

INPUT=$(cat 2>/dev/null || true)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty' 2>/dev/null || true)

[ -z "$FILE_PATH" ] && exit 0

echo "$FILE_PATH" >> "$CLAUDE_PROJECT_DIR/.claude/.read-tracker"
exit 0
