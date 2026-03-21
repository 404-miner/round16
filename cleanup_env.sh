#!/bin/bash
set -e

ENV_NAME="404-base-miner-env"

# Initialize conda
if [ -n "$(which conda)" ]; then
    CONDA_BASE=$(conda info --base)
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
    echo "Conda not found"
    exit 1
fi

# Deactivate if active
if [[ "$CONDA_DEFAULT_ENV" == "$ENV_NAME" ]]; then
    conda deactivate
fi

# Remove environment
if conda env list | grep -q "^${ENV_NAME}\s"; then
    echo "Removing $ENV_NAME environment..."
    conda env remove -n "$ENV_NAME" -y
    echo "Done."
else
    echo "Environment $ENV_NAME does not exist."
fi