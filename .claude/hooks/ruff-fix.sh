#!/usr/bin/env bash
# PostToolUse hook: format and autofix any Python file Claude writes.
#
# This is the same ruff-format + ruff-check --fix pair that .pre-commit-config.yaml
# runs, applied one file at a time as it is edited. The point is not the formatting
#, pre-commit would catch that anyway. It is that whatever ruff *cannot* fix gets
# reported straight back into the conversation, so lint errors are corrected while
# the code is still in context rather than at commit time.
#
# Exits 0 unconditionally. A formatter that can block an edit is a formatter that
# will eventually block the wrong edit.
set -uo pipefail

payload="$(cat)"
file="$(printf '%s' "$payload" | jq -r '.tool_response.filePath // .tool_input.file_path // empty')"

case "$file" in
*.py) ;;
*) exit 0 ;;
esac
[ -f "$file" ] || exit 0

root="${CLAUDE_PROJECT_DIR:-$(git -C "$(dirname "$file")" rev-parse --show-toplevel 2>/dev/null)}"
[ -n "$root" ] || exit 0

# Prefer the venv binary (~25 ms). `uv run` is the fallback for a fresh clone that
# has not been synced yet; if neither exists there is nothing useful to do.
if [ -x "$root/.venv/bin/ruff" ]; then
	ruff() { "$root/.venv/bin/ruff" "$@"; }
elif command -v uv >/dev/null 2>&1; then
	ruff() { uv run --project "$root" --quiet ruff "$@"; }
else
	exit 0
fi

ruff format --quiet "$file" >/dev/null 2>&1
ruff check --quiet --fix "$file" >/dev/null 2>&1

# --quiet suppresses the "All checks passed!" summary, which is otherwise non-empty
# stdout and would be reported back as a finding on every clean edit.
remaining="$(ruff check --no-cache --quiet --output-format=concise "$file" 2>/dev/null)"
if [ -n "$remaining" ]; then
	jq -n --arg r "$remaining" '{
		hookSpecificOutput: {
			hookEventName: "PostToolUse",
			additionalContext: ("ruff reported issues it could not autofix. Fix them now:\n" + $r)
		}
	}'
fi

exit 0
