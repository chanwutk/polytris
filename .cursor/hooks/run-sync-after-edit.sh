#!/usr/bin/env bash
# Cursor afterFileEdit hook: sync workspace to remote after agent file edits.
# stdin: hook JSON payload (must be consumed)

cat >/dev/null

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 0

exec ./sync
