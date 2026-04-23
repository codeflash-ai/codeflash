#!/usr/bin/env bash
set -euo pipefail

input=$(cat)
file_path=$(echo "$input" | jq -r '.tool_input.file_path // empty')

if [[ -z "$file_path" || ! -f "$file_path" ]]; then
    exit 0
fi

if [[ "$file_path" == *.py ]]; then
    uv run prek --files "$file_path" 2>/dev/null || uv run prek --files "$file_path"
fi
