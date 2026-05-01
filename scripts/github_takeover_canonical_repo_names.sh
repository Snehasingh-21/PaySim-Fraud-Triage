#!/usr/bin/env bash
set -euo pipefail

# Deletes the OLD polluted repos on GitHub, then renames the clean "-fresh" repos to
# the original names so URLs and portfolio links stay stable.
#
# Prereq (one-time):
#   gh auth refresh -h github.com -s delete_repo
#
# Requires: gh CLI, HTTPS auth with delete_repo + repo scopes.

OWNER="Snehasingh-21"

echo "Deleting old repos (need delete_repo scope)..."
gh repo delete "${OWNER}/PaySim-Fraud-Triage" --yes
gh repo delete "${OWNER}/multi-coin-crypto-analytics-using-coingecko-api" --yes

echo "Renaming clean repos → canonical names..."
gh repo rename "PaySim-Fraud-Triage" --repo "${OWNER}/paysim-fraud-triage-fresh"
gh repo rename "multi-coin-crypto-analytics-using-coingecko-api" --repo "${OWNER}/multicoin-coingecko-fresh"

echo "Done. Canonical URLs:"
echo "  https://github.com/${OWNER}/PaySim-Fraud-Triage"
echo "  https://github.com/${OWNER}/multi-coin-crypto-analytics-using-coingecko-api"

if [[ -d "$(git rev-parse --show-toplevel 2>/dev/null)" ]]; then
  ROOT="$(git rev-parse --show-toplevel)"
  if git -C "$ROOT" remote get-url origin &>/dev/null; then
    echo "Pointing this clone origin at canonical PaySim URL..."
    git -C "$ROOT" remote set-url origin "https://github.com/${OWNER}/PaySim-Fraud-Triage.git"
    git -C "$ROOT" fetch origin
    git -C "$ROOT" reset --hard "origin/main" || git -C "$ROOT" checkout main && git -C "$ROOT" reset --hard "origin/main"
  fi
fi
