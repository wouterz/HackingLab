#!/bin/sh

#SBATCH --partition=student
#SBATCH --qos=student
#SBATCH --time=0:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4096
#SBATCH --mail-type=END
#SBATCH --gres=gpu

srun python3 ./randomAdvScript.py
