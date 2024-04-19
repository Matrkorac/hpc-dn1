#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=seam_carve
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=seam_carve_out.log

export OMP_PLACES=cores
export OMP_PROC_BIND=TRUE
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

gcc -fopenmp -lm carve_simple.c -o seam_carve

# run the program with input parameters
# srun seam_carve ../test_images/720x480.png ./720x480_out.png 16 
# srun seam_carve ../test_images/720x480.png ./720x480_out.png 32 
# srun seam_carve ../test_images/720x480.png ./720x480_out.png 64 
# srun seam_carve ../test_images/720x480.png ./720x480_out.png 128 

#srun seam_carve ../test_images/1024x768.png ./1024x768_out.png 16 
#srun seam_carve ../test_images/1024x768.png ./1024x768_out.png 32 
#srun seam_carve ../test_images/1024x768.png ./1024x768_out.png 64 
# srun seam_carve ../test_images/1024x768.png ./1024x768_out.png 128 

#srun seam_carve ../test_images/1920x1200.png ./1920x1200_out.png 16
#srun seam_carve ../test_images/1920x1200.png ./1920x1200_out.png 32
#srun seam_carve ../test_images/1920x1200.png ./1920x1200_out.png 64
# srun seam_carve ../test_images/1920x1200.png ./1920x1200_out.png 128

#srun seam_carve ../test_images/3840x2160.png ./3840x2160_out.png 16
#srun seam_carve ../test_images/3840x2160.png ./3840x2160_out.png 32
#srun seam_carve ../test_images/3840x2160.png ./3840x2160_out.png 64
srun seam_carve ../test_images/3840x2160.png ./3840x2160_out.png 128

#srun seam_carve ../test_images/7680x4320.png .7680x4320_out.png 16
#srun seam_carve ../test_images/7680x4320.png .7680x4320_out.png 32
#srun seam_carve ../test_images/7680x4320.png .7680x4320_out.png 64
#srun seam_carve ../test_images/7680x4320.png .7680x4320_out.png 128