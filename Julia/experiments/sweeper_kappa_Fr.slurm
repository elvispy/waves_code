#!/bin/bash
#SBATCH --job-name=SB_kappa_Fr
#SBATCH --output=../logs/kappa_Fr_%a.out
#SBATCH --error=../logs/kappa_Fr_%a.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=1-100

# Inviscid (κ, Fr) sweep at fixed motor position. One array task per Fr slice
# (each task does all κ values at a single Fr). Match --array to nFr in the
# Julia script (default 100).

cd "$(dirname "$(realpath "$0")")"

module load julia

julia --project=../ sweeper_kappa_Fr.jl $SLURM_ARRAY_TASK_ID both
