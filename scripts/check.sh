# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

set -e
set -o xtrace

if [ ! -z "${POPLAR_SDK_ENABLED}" ]; then
    ninja -f ipu/build.ninja
    python -m pytest ipu/test.py
fi

python -m pytest tests/

python -m black --check src/ scripts/ tests/

python -m isort --check src/ scripts/ tests/
