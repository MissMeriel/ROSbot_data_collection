#!/bin/bash

module load python-3.8.0

# Specify the path to the Python 3.8 executable
PYTHON_EXEC="sw/ubuntu-22.04/python-3.8.0/bin/python3"

# Check if the specified Python version is correct
pythonversion="$($PYTHON_EXEC --version 2>&1)"
if echo "$pythonversion" | grep -qE "Python 3\.8|Python 3\.9"; then
  echo "Using Python version $pythonversion"
else
  echo "Python version $pythonversion is not compatible. Use Python 3.8 or 3.9"
  exit 1
fi

# Create virtual environment
$PYTHON_EXEC -m venv .venv
. .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install torch torchvision numpy black mypy scipy scikit-image pandas opencv-python matplotlib kornia

# Create datasets directory and download dataset
mkdir -p ../datasets
wget -O ../datasets/dataset.zip "https://virginia.box.com/shared/static/fb9lj05cg6twkq92gh7el3q9jdu5wd1n"
unzip ../datasets/dataset.zip -d ../datasets

echo "Setup complete"
