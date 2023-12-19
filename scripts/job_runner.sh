#!/usr/bin/env bash
# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

SCRIPT=$1

VENV="/tmp/venv-${SLURM_JOB_ID:-$(date --iso-8601=seconds)}"

set -e
trap "rm -rf ${VENV}" EXIT

python -m venv "${VENV}"
echo "export TOKENIZERS_PARALLELISM=true" >> "${VENV}/bin/activate"
source "${VENV}/bin/activate"
pip install wheel
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install .

echo -e "\n\nRUNNING ${SCRIPT}"
python "${SCRIPT}"
