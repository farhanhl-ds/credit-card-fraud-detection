#!/usr/bin/env bash
# Downloads the Kaggle Playground Series S6E11 data into ./data
#
# One-time setup:
#   1. Create a Kaggle API token (kaggle.com > Settings > API) and save it as
#      ~/.kaggle/kaggle.json (or export KAGGLE_API_TOKEN).
#   2. Open https://www.kaggle.com/competitions/playground-series-s6e11 and accept the rules.
#   3. pip install kaggle (already included in requirements.txt)
set -euo pipefail

mkdir -p data
kaggle competitions download -c playground-series-s6e11 -p data
unzip -o data/playground-series-s6e11.zip -d data
ls -lh data
