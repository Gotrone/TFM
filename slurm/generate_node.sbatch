#!/bin/bash
#SBATCH --job-name=cfdbench-gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=27       # one core per case (27 cases)
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out

# module load python/3.12        # adapt to your cluster
source .venv/bin/activate
export OMP_NUM_THREADS=1         # one thread per process: parallelism comes from the processes
cfdbench generate --workers "$SLURM_CPUS_PER_TASK"
