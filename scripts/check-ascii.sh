#!/usr/bin/env bash
# Fail if any tracked C++/CMake source file contains non-ASCII bytes.
# Keeps sources copy-paste-safe across editors, terminals, and toolchains.
set -euo pipefail
cd "$(dirname "$0")/.."

status=0
while IFS= read -r f; do
  if LC_ALL=C grep -nP '[^\x00-\x7F]' -- "$f" >/dev/null 2>&1; then
    echo "non-ASCII byte(s) in $f:"
    LC_ALL=C grep -nP '[^\x00-\x7F]' -- "$f" || true
    status=1
  fi
done < <(git ls-files '*.hpp' '*.cpp' '*.cu' '*.cuh' '*.cmake' '*CMakeLists.txt')

if [ "$status" -ne 0 ]; then
  echo "ascii lint failed: replace the bytes above with ASCII equivalents." >&2
fi
exit "$status"
