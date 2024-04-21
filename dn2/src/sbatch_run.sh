#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=histogram_serial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=histogram_serial_out.log

export OMP_PLACES=cores
export OMP_PROC_BIND=TRUE
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

FILE=histogram_serial

gcc -fopenmp -lm ${FILE}.c -o ${FILE}

# run the program with input parameters
srun ${FILE} ../../test_images/720x480.png ../results/720x480_out.png

# srun histogram_serial ../../test_images/1024x768.png ../results/1024x768_out.png

#srun histogram_serial ../../test_images/1920x1200.png ../results/1920x1200_out.png

#srun histogram_serial ../../test_images/3840x2160.png ../results/3840x2160_out.png

#srun histogram_serial ../../test_images/7680x4320.png ../results/7680x4320_out.png

rm ./${FILE}