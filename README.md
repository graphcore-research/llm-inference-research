# SparQ attention

Code to support [SparQ Attention](https://arxiv.org/abs/2312.04985).

 - This tag contains the evaluation framework "LLM inference research" and SparQ implementations, as used for the main experiments (see [main](https://github.com/graphcore-research/llm-inference-research) for the most recent version).
 - [notebooks/2023-sparse-attention](https://github.com/graphcore-research/llm-inference-research/tree/notebooks/2023-sparse-attention) includes source for the figures and analysis
 - [benchmarks](https://github.com/graphcore-research/llm-inference-research/tree/benchmarks) implements the microbenchmarks

The following README is the generic description of the evaluation framework.

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
# On a CPU-only machine, you may need to run this before `pip install -r requirements.txt`
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
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
