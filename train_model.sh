#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH -n 4
#SBATCH --mem=60G
#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH -J timm-model
#SBATCH -o logs/timm-model-%j.out

cd /gpfs/data/tserre/jgopal/Clickme_Modeling/

source timm.venv/bin/activate

python3 train_model.py


