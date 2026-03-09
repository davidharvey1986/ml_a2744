#!/usr/bin/env bash

CWD=`echo $PWD`
set -e

ENV_NAME="astro-env"

# Create virtual environment

python3.11 -m venv $ENV_NAME

# Activate it

source $ENV_NAME/bin/activate

# Upgrade pip

pip install --upgrade pip setuptools==59.8.0 wheel

# Core scientific stack

pip install numpy==1.26.1 scipy pandas  matplotlib seaborn scikit-learn ipython ipykernel jupyter

# PyTorch (CUDA 12.6 wheels)

pip install torch torchvision

# Utilities

pip install wandb gdown sympy networkx filelock jinja2 fsspec requests pyyaml pillow lenspack

if [ ! -d pyRRG]
then
    git clone git@github.com:davidharvey1986/pyRRG.git
fi

cd pyRRG 
pip install .
cd ..

echo "export PYTHONPATH=${CWD}" >> $ENV_NAME/bin/activate

echo ""
echo "Environment created."
echo "Activate it with:"
echo "source $ENV_NAME/bin/activate"
