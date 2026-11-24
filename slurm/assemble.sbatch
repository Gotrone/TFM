#!/bin/bash
#SBATCH --job-name=cfdbench-assemble
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out

# module load python/3.12
source .venv/bin/activate
cfdbench assemble --cases data/cases --output data/cylinder_flow.npz
cfdbench benchmark --reset
