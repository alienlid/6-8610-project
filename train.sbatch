#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --output=imdb-bert.out
#SBATCH --array=0
srun python3 train.py

