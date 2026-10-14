#!/usr/bin/env bash
# Downloads the Kaggle (Ingle 2020) crop recommendation CSV from a public GitHub mirror.
set -euo pipefail
cd "$(dirname "$0")"
curl -fsSL -o Crop_recommendation.csv \
  "https://raw.githubusercontent.com/Gladiator07/Harvestify/master/Data-processed/crop_recommendation.csv"
echo "saved $(wc -l < Crop_recommendation.csv) lines to $(pwd)/Crop_recommendation.csv"
