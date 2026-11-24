#!/bin/bash
#SBATCH --job-name=cfdbench-case
#SBATCH --array=0-26             # 27 cases, see `cfdbench info` (train 6 + interp 15 + extrap 6)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%A_%a.out

# module load python/3.12
source .venv/bin/activate
export OMP_NUM_THREADS=1
cfdbench generate-case --out data/cases     # case index = $SLURM_ARRAY_TASK_ID
