# Running on a SLURM cluster

Three ways to generate the dataset (27 simulations). **These scripts were not run on a cluster**: I had no cluster access. The commands they call (`generate`, `generate-case`, `assemble`) were run and tested locally. Adapt the `module load` lines and the partition names to your site.

| Script | What it does | When to use it |
|---|---|---|
| `generate_node.sbatch` | one job, all cases in parallel on one node (`--workers $SLURM_CPUS_PER_TASK`) | simplest, a single big node |
| `generate_array.sbatch` | one array task per case, then `assemble.sbatch` | many small nodes, or cases that must run independently |
| `generate_gpu.sbatch` | one job on one GPU | when a GPU node is available (see the grid-size study: only faster for large grids) |

```bash
# array version: submit the 27 cases, then the assembly once they all succeeded
sbatch --parsable slurm/generate_array.sbatch            # prints the job id, say 12345
sbatch --dependency=afterok:12345 slurm/assemble.sbatch
```

Environment setup (once):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                  # for the GPU script also: pip install "cupy-cuda12x[ctk]"
```
