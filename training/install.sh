#!/usr/bin/bash
pythonversion="$(python --version)"

if echo "$pythonversion"* | grep -iq "Python 3.8" || echo "$pythonversion"* | grep -iq "Python 3.9" ; then
  echo "Using Python version $pythonversion"
else
  echo "Python version $pythonversion is not compatible. Use python 3.8 or 3.9"
  exit
fi
python -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision numpy black mypy scipy scikit-image pandas opencv-python matplotlib kornia
mkdir ../datasets
wget -O ../datasets "https://virginia.box.com/shared/static/fb9lj05cg6twkq92gh7el3q9jdu5wd1n"