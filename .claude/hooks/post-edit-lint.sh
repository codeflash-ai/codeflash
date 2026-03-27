#!/usr/bin/env bash
# Everyone is on macOS so this should be fine, we don't account for Windows
set -euo pipefail

input=$(cat)
file_path=$(echo "$input" | jq -r '.tool_input.file_path // empty')

if [[ -z "$file_path" || ! -f "$file_path" ]]; then
    exit 0
fi

if [[ "$file_path" == *.py ]]; then
    uv run prek --files "$file_path" 2>/dev/null || true
fi
