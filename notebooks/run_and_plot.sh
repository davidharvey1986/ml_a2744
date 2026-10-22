#!/bin/bash
# run_and_plot.sh  –  run from ml_check/notebooks/
# Uses the project's own virtualenv Python on macOS

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON="$SCRIPT_DIR/../astro-env/bin/python3"

if [ ! -f "$PYTHON" ]; then
    echo "ERROR: virtualenv python not found at $PYTHON"
    exit 1
fi

cd "$SCRIPT_DIR"
"$PYTHON" run_and_plot.py "$@"
