#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --reservation=fri
#SBATCH --job-name=code_sample
#SBATCH --gpus=1
#SBATCH --output=sample_out.log

FILE=sample

# module keyword cuda

# module load CUDA/11.0.2-GCC-9.3.07
# module load CUDA

module load CUDAcore/11.2.1

nvcc -diag-suppress 550 -O2 -lm ${FILE}.cu -o ${FILE}
# nvcc  -O2 -lm ${FILE}.cu -o ${FILE}

./${FILE} ../../test_images/1024x768.png ../results/1024x768_out.png

#srun ${FILE} ../../test_images/1920x1200.png ../results/1920x1200_out.png

#srun ${FILE} ../../test_images/3840x2160.png ../results/3840x2160_out.png

#srun ${FILE} ../../test_images/7680x4320.png ../results/7680x4320_out.png

rm ${FILE}