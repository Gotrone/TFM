#!/bin/bash
#SBATCH --job-name=cfdbench-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out

# module load cuda/12.4          # adapt to your cluster
source .venv/bin/activate
cfdbench generate --backend gpu --dtype float64
