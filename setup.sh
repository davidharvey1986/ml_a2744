#!/usr/bin/env bash

# Create environment

conda create -y -n astro-env python=3.9

# Activate environment

source $(conda info --base)/etc/profile.d/conda.sh
conda activate astro-env

# Install core scientific stack

conda install -y 
numpy 
scipy 
pandas 
matplotlib 
seaborn 
scikit-learn 
ipython 
ipykernel 
jupyter 
pip

# Install ML / astronomy related packages

pip install 
torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install 
wandb 
gdown 
sympy 
networkx 
filelock 
jinja2 
fsspec 
requests 
pyyaml 
pillow 
pyrrg

echo "Environment astro-env created successfully."
echo "Activate it with: conda activate astro-env"
