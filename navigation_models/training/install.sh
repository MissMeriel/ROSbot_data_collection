#!/bin/bash
. ../../.venv-rb/bin/activate
PYTHON_EXEC="$(readlink -f $(which python3))"

# Check if the specified Python version is correct
pythonversion="$($PYTHON_EXEC --version 2>&1)"
if echo "$pythonversion" | grep -qE "Python 3\.9|Python 3\.10"; then
  echo "Using Python version $pythonversion"
else
  echo "Python version $pythonversion is not compatible. Use Python 3.9 or 3.10"
  exit 1
fi

# Upgrade pip and install dependencies
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision numpy black mypy scipy scikit-image pandas opencv-python matplotlib kornia

# Create datasets directory and download dataset
ls ../../dataset/
if [ ! -d ../../dataset/ ]; then
  echo "did not find training dataset in parent directory, downloading to ../../dataset"
  gdown --folder  https://drive.google.com/drive/folders/1Zn7ZNDpPpw7ffnotwR8Jb-DGITRNZy0A -O ../../
  for file in ../../dataset/*; do
     if (file $file | grep -q compression ) ; then
       echo "found $file, unzipping to ${file%.*}"
       unzip $file -d ../../dataset/
     fi
  done
fi
echo "Setup complete"
