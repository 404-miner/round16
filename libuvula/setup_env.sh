#!/bin/bash

# Stop the script on any error
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_FILE="${SCRIPT_DIR}/conda_env.yml"
WORKDIR=$(pwd)

# Attempt to find Conda's base directory and source it (required for `conda activate`)
export PATH="$HOME/miniconda3/bin:$PATH"
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Create environment and activate it
#conda env create -f conda_env.yml
conda activate libuvula-env
conda info --env

# Create build folder
mkdir -p build && cd build

# Configure CMake
cmake .. \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
    -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    -DCMAKE_BUILD_TYPE=Release \
    -DUVULA_BUILD_PYTHON_BINDINGS=ON \
    -DUVULA_BUILD_JS_BINDINGS=OFF \
    -DUVULA_BUILD_CLI=OFF \
    -DUVULA_ENABLE_EXTENSIVE_WARNINGS=OFF

cmake --build . -- -j$(nproc)

cd ..
pip install build
python -m build -w -o "./wheels"

echo "✅ Uvula built successfully!"

