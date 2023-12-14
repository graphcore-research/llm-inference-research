# Research LLM inference

## Setup

See [notebooks:Eval.ipynb](https://github.com/PRODUCT-AI-ENGINEERING-GCAI/research-llm-inference/blob/notebooks/Eval.ipynb) for usage.

```bash
python3 -m venv .venv
# Append to .venv/bin/activate:
    export PYTHONPATH="${PYTHONPATH}:$(dirname ${VIRTUAL_ENV})"
    export TOKENIZERS_PARALLELISM=true

source .venv/bin/activate
pip install wheel
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Optional - notebooks
git clone git@github.com:PRODUCT-AI-ENGINEERING-GCAI/research-llm-inference.git --branch notebooks notebooks/
```
