#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=seam_carving
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=seam_carve_out.log

export OMP_PLACES=cores
export OMP_PROC_BIND=TRUE
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

gcc -O2 -lm --openmp seam_carve.c -o seam_carve

# run the program with input parameters
srun seam_carve valve.png valve_out.png 