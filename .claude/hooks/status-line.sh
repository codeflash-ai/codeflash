#!/usr/bin/env bash
# Status line: derive context from git state.

input=$(cat)
project_dir=$(echo "$input" | jq -r '.workspace.project_dir')

user=$(whoami)
branch=$(git -C "$project_dir" branch --show-current 2>/dev/null)

changed=$(git -C "$project_dir" diff --name-only HEAD 2>/dev/null)
[ -z "$changed" ] && changed=$(git -C "$project_dir" diff --name-only 2>/dev/null)
[ -z "$changed" ] && changed=$(git -C "$project_dir" diff --name-only --cached 2>/dev/null)

if [ -n "$changed" ]; then
    area=$(echo "$changed" | sed 's|/.*||' | sort | uniq -c | sort -rn | head -1 | awk '{print $2}')
else
    area=""
fi

context=""
case "$area" in
    codeflash)
        subsystem=$(echo "$changed" | grep '^codeflash/' | sed 's|^codeflash/||; s|/.*||' | sort | uniq -c | sort -rn | head -1 | awk '{print $2}')
        [ -n "$subsystem" ] && context="editing $subsystem" ;;
    tests)
        target=$(echo "$changed" | grep '^tests/' | sed 's|^tests/||; s|/.*||' | sort -u | head -1)
        [ -n "$target" ] && context="testing $target" ;;
    .claude)
        context="configuring claude" ;;
esac

if [ -z "$context" ] && [ -n "$branch" ]; then
    case "$branch" in
        feat/*|cf-*) context="building: ${branch#feat/}" ;;
        fix/*)       context="fixing: ${branch#fix/}" ;;
        refactor/*)  context="refactoring: ${branch#refactor/}" ;;
        test/*)      context="testing: ${branch#test/}" ;;
        chore/*)     context="chore: ${branch#chore/}" ;;
    esac
fi

dirty=""
if [ -n "$(git -C "$project_dir" status --porcelain 2>/dev/null)" ]; then
    dirty=" *"
fi

status="$user | codeflash"
[ -n "$context" ] && status="$status | $context"
[ -n "$branch" ] && status="$status | $branch$dirty"
echo "$status"
