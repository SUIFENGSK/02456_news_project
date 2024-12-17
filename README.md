# Introduction
The project is based on the original repository [ebnerd-benchmark](https://github.com/ebanalyse/ebnerd-benchmark) and has been reimplemented in PyTorch.

# Getting Started
To run the program, you need to install the necessary packages:

```
# 1. Create and activate a new environment
module load python3/3.10.14
python -m venv /path/to/new/virtual/environment
cd my/project/path
python3 -m venv .venv
source /path/to/new/virtual/environment/bin/activate

# 2. Install the core ebrec package to the enviroment:
pip install .
```

# Running on DTU HPC
If you want to run the model on the DTU HPC, the following commands are suggested:
```
# 1. Choose sxm2sh GPU nodes 
sxm2sh

# 2. Load python and activate environment
module load python3/3.10.14
source .venv/bin/activate

# 3. Load cuda and set GPU devices
module load cuda/12.2
export CUDA_VISIBLE_DEVICES=0,1,2,3
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/appl/cuda/12.2.0

# 4. Run the script
python3 02456_news_project/src/torch_nerd/main.py
```

# Source Code
The entire source code can be found under ```src/torch_nerd```. Other code, including the ```from_ebrec``` package, is from the original repository.

# Notebook
The notebook, located at ```src/torch_nerd/main.ipynb```, can be used to reproduce our results. The parameters specified in the notebook represent the final version we used to train our model, but they can be modified as desired.