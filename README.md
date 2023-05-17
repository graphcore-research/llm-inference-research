# Research LLM inference

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Patch & install Eleuther's LM harness
mkdir -p third_party
git clone https://github.com/EleutherAI/lm-evaluation-harness.git third_party/lm-evaluation-harness
git -C third_party/lm-evaluation-harness reset --hard d145167
git -C third_party/lm-evaluation-harness apply ../lm-evaluation-harness.patch
pip install -e third_party/lm-evaluation-harness/
```
