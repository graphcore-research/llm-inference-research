# SparQ benchmarks

```sh
python3 -m venv .venv
# Append to .venv/bin/activate
#   export PYTHONPATH="${PYTHONPATH}:$(dirname ${VIRTUAL_ENV})/src"
source .venv/bin/activate
pip install wheel
pip install -r requirements.txt

./scripts/check.sh

# IPU Sweep
python scripts/run_sweep_ipu.py

# CPU/GPU Sweep
python scripts/run_sweep_pytorch.py
```

## End-to-end experiments

* Run SparQ on CPU using `llama.cpp` [here](https://github.com/graphcore-research/sparq-llama.cpp)
* Run SparQ on GPU using `gpt-fast` [here](https://github.com/graphcore-research/sparq-gpt-fast)

## License

Copyright (c) 2024 Graphcore Ltd. Licensed under the MIT License (see [LICENSE](LICENSE)).

Our dependencies are documented in [requirements.txt](requirements.txt).
