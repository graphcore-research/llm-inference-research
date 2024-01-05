# SparQ benchmarks

```sh
python3 -m venv .venv
# Append to .venv/bin/activate
#   export PYTHONPATH="${PYTHONPATH}:$(dirname ${VIRTUAL_ENV})/src"
source .venv/bin/activate
pip install wheel
pip install -r requirements.txt
./scripts/check.sh
```
