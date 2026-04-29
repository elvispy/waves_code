#!/bin/bash
#SBATCH --job-name=surferbot_array
#SBATCH --output=../logs/sweep_%a.out
#SBATCH --error=../logs/sweep_%a.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1  # Setting to 1 since we parallelize via Array Tasks now
#SBATCH --mem=8G           # 8GB is plenty for one EI row (100 sims)
#SBATCH --array=1-300      # One task for each EI value

module load julia

# Pass the SLURM_ARRAY_TASK_ID to the script
julia --project=../ brute_force_sweep.jl $SLURM_ARRAY_TASK_ID
