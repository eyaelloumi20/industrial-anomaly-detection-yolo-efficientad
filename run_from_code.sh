#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT/code"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
if [[ $# -lt 1 ]]; then
    echo "Usage: ./run_from_code.sh <original_script.py|original_script.sh> [args...]" >&2
    exit 2
fi
script="$1"; shift
case "$script" in
    /*|../*|*/../*|*/..|.. ) echo "Use relative paths inside code/ only" >&2; exit 2;;
esac
if [[ ! -f "$script" ]]; then
    echo "Missing original script: $script" >&2
    exit 2
fi
case "$script" in
    *.py) exec python3 "$script" "$@";;
    *.sh) exec bash "$script" "$@";;
    *) echo "Expected a .py or .sh file" >&2; exit 2;;
esac
