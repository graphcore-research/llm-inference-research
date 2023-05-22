# Research LLM inference

## Setup

See [notebooks:Eval.ipynb](https://github.com/PRODUCT-AI-ENGINEERING-GCAI/research-llm-inference/blob/notebooks/Eval.ipynb) for usage.

```bash
python3 -m venv .venv
# Append to .venv/bin/activate:
    export PYTHONPATH="${PYTHONPATH}:$(dirname ${VIRTUAL_ENV})"
    export TOKENIZERS_PARALLELISM=true

source .venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Patch & install Eleuther's LM harness
mkdir -p third_party
git clone https://github.com/EleutherAI/lm-evaluation-harness.git third_party/lm-evaluation-harness
git -C third_party/lm-evaluation-harness reset --hard d145167
git -C third_party/lm-evaluation-harness apply ../lm-evaluation-harness.patch
pip install -e third_party/lm-evaluation-harness/

# Optional - reference data for `llminference.outcompare`
git clone git@github.com:PRODUCT-AI-ENGINEERING-GCAI/research-llm-inference.git --branch data data/

# Optional - notebooks
git clone git@github.com:PRODUCT-AI-ENGINEERING-GCAI/research-llm-inference.git --branch notebooks notebooks/
```
