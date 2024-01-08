set -e
set -o xtrace

pytest tests/

black --check src/ scripts/ tests/

isort --check src/ scripts/ tests/
