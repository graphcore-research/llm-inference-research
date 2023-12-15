# LLM inference research

Experimentation framework from Graphcore Research, used to explore the machine learning performance of post-training model adaptation for accelerating LLM inference.

See: [SparQ Attention](https://arxiv.org/abs/2312.04985).

## Setup

See [scripts/Eval.ipynb](scripts/Eval.ipynb) and [scripts/Quantisation.ipynb](scripts/Quantisation.ipynb) for usage.

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

## Development

We use a script called `dev` to automate building, testing, etc.

```bash
./dev
./dev --help
```

## License

Copyright (c) 2023 Graphcore Ltd. Licensed under the MIT License.

See [NOTICE.md](NOTICE.md) for further details.
