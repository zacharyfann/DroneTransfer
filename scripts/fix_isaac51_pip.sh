#!/usr/bin/env bash
# Repair Isaac Sim 5.1 Python env after a broken pip install.
# Fixes: torch 2.12 upgrade, missing prettytable/warp, wrong gymnasium pin.
#
# Run INSIDE isaac-sim-51 as root:
#   bash /workspace/drone/scripts/fix_isaac51_pip.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/setup_isaaclab_51_main.sh"
