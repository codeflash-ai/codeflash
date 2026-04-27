# Session Discipline

## Scope

One task per session. Don't mix implementation with communication drafting, transcript search, or strategic planning.

## Duration

Cap sessions at 2-3 hours. Use `/handoff` at natural breakpoints rather than letting auto-compaction degrade context.

- After 1 compaction: consider wrapping up the current task and handing off
- After 3 compactions: stop, and tell the user to start a fresh session
- Never continue past 5 compactions — context is too degraded

## Context preservation

When compacting, preserve: modified files list, current branch, test commands used, key decisions made. Use subagents for exploration to keep main context clean.

## No polling

Never poll background tasks. No `wc -l`, no `tail -f`, no `sleep` loops. Use `run_in_background` and wait for the completion notification.

## File read budget

If you've read the same file 3+ times in a session, either:
- The session is too long and compaction destroyed your context — write a handoff
- You're not retaining key information — write it down in your response before it compacts away
