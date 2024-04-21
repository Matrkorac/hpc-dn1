#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include "helper_cuda.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define COLOR_CHANNELS 0


__global__ void compute_histograms(const unsigned char *image,
                                    const int width, const int height, const int cpp,
                                    int Hr[256], int Hg[256], int Hb[256])
{

    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    // compute the histograms for all three channels
    if (row == 0 & col == 0)
    {
        printf("DEVICE: START HISTOGRAM EQUALIZATION\n");
    }

    // memory shared within each block
    __shared__ int Hrs[256];
    __shared__ int Hgs[256];
    __shared__ int Hbs[256];

    for (int i = row; i < height; i += blockDim.x * gridDim.x)
    {
        for (int j = col; j < width; j += blockDim.y * gridDim.y)
        {
            atomicAdd(&Hrs[image[(i * width + j) * cpp + 0]], 1);
            atomicAdd(&Hgs[image[(i * width + j) * cpp + 1]], 1);
            atomicAdd(&Hbs[image[(i * width + j) * cpp + 2]], 1);
            // imageOut[(i * width + j) * cpp + c] = image[(i * width + j) * cpp + c];
        }
    }

    __syncthreads();

    // each thread adds a single value into global memory for each histogram
    atomicAdd(&Hr[threadIdx.x * blockDim.y + threadIdx.y], Hrs[threadIdx.x * blockDim.y + threadIdx.y]);
    atomicAdd(&Hg[threadIdx.x * blockDim.y + threadIdx.y], Hgs[threadIdx.x * blockDim.y + threadIdx.y]);
    atomicAdd(&Hb[threadIdx.x * blockDim.y + threadIdx.y], Hbs[threadIdx.x * blockDim.y + threadIdx.y]);
}


// naive implemetation with only three threads
__global__ void cumulative_histograms(int Hr[256], int Hg[256], int Hb[256], int *min_h)
{
    int idx = threadIdx.x;

    if (idx == 0) {
        int min_r = Hr[0];
        for (int i = 1; i < 256; i++) {
            Hr[i] = Hr[i] + Hr[i-1];
            if (min_r == 0 && Hr[i] != 0)
                min_r = Hr[i];
        }
        min_h[0] = min_r;
    }

    if (idx == 1) {
        int min_g = Hg[0];
        for (int i = 1; i < 256; i++) {
            Hg[i] = Hg[i] + Hg[i-1];
            if (min_g == 0 && Hg[i] != 0) 
                min_g = Hg[i];
        }
        min_h[1] = min_g;
    }

    if (idx == 2) {
        int min_b = Hb[0];
        for (int i = 1; i < 256; i++) {
            Hb[i] = Hb[i] + Hb[i-1];
            if (min_b == 0 && Hb[i] != 0) 
                min_b = Hb[i];
        }
        min_h[2] = min_b;
    }
}


// can be completely parallelized
__global__ void new_intensities(int Hr[256], int Hg[256], int Hb[256],
                                unsigned char new_int_r[256],
                                unsigned char new_int_g[256],
                                unsigned char new_int_b[256],
                                int* min_h,
                                int N, int M, int L)
{
    // can have three blocks; one for each color channel
    // one block of 256 threads (16x16)
    int idx = blockIdx.x * threadIdx.y * blockDim.x + threadIdx.x;

    if (blockIdx.x == 0) 
        new_int_r[idx] = (unsigned char) (((Hr[idx] - min_h[0]) / (N * M * min_h[0])) * (L - 1));

    if (blockIdx.x == 1) 
        new_int_g[idx] = (unsigned char) (((Hg[idx] - min_h[1]) / (N * M * min_h[1])) * (L - 1));

    if (blockIdx.x == 2) 
        new_int_b[idx] = (unsigned char) (((Hb[idx] - min_h[2]) / (N * M * min_h[2])) * (L - 1));
}


// can be completely parallelized
__global__ void assign_intensities(unsigned char *image, int height, int width, int cpp,
                                    unsigned char int_r[256],
                                    unsigned char int_g[256],
                                    unsigned char int_b[256])
{
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;

    for (int i = gidx; i < height; i += blockDim.x * gridDim.x)
    {
        for (int j = gidy; j < width; j += blockDim.y * gridDim.y)
        {
            image[(i * width + j) * cpp + 0] = int_r[image[(i * width + j) * cpp + 0]];
            image[(i * width + j) * cpp + 1] = int_r[image[(i * width + j) * cpp + 1]];
            image[(i * width + j) * cpp + 2] = int_r[image[(i * width + j) * cpp + 2]];
        }
    }
}


int main(int argc, char *argv[])
{

    if (argc < 3)
    {
        printf("USAGE: sample input_image output_image\n");
        exit(EXIT_FAILURE);
    }

    char szImage_in_name[255];
    char szImage_out_name[255];

    snprintf(szImage_in_name, 255, "%s", argv[1]);
    snprintf(szImage_out_name, 255, "%s", argv[2]);

    // Load image from file and allocate space for the output image
    int width, height, cpp;
    unsigned char *h_imageIn = stbi_load(szImage_in_name, &width, &height, &cpp, COLOR_CHANNELS);

    if (h_imageIn == NULL)
    {
        printf("Error reading loading image %s!\n", szImage_in_name);
        exit(EXIT_FAILURE);
    }
    printf("Loaded image %s of size %dx%d.\n", szImage_in_name, width, height);
    const size_t datasize = width * height * cpp * sizeof(unsigned char);
    // unsigned char *h_imageOut = (unsigned char *)malloc(datasize);

    //dim3 gridSize(1, 1);

    unsigned char *d_imageIn;
    // unsigned char *d_imageOut;

    int *H_r;
    int *H_g;
    int *H_b;
    const size_t histogram_size = 256 * sizeof(int);

    unsigned char *newIntR;
    unsigned char *newIntG;
    unsigned char *newIntB;
    const size_t intensities_size = 256 * sizeof(unsigned char);

    int *min_h;
    const size_t min_h_size = 3 * sizeof(int);

    // Allocate memory on the device
    checkCudaErrors(cudaMalloc(&d_imageIn, datasize));
    // checkCudaErrors(cudaMalloc(&d_imageOut, datasize));

    // allocate the histograms for color channels to the device
    checkCudaErrors(cudaMalloc(&H_r, histogram_size));
    checkCudaErrors(cudaMalloc(&H_g, histogram_size));
    checkCudaErrors(cudaMalloc(&H_b, histogram_size));

    checkCudaErrors(cudaMalloc(&newIntR, intensities_size));
    checkCudaErrors(cudaMalloc(&newIntG, intensities_size));
    checkCudaErrors(cudaMalloc(&newIntB, intensities_size));

    checkCudaErrors(cudaMalloc(&min_h, min_h_size));

    // Use CUDA events to measure execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Copy image to device and run kernel
    cudaEventRecord(start);
    checkCudaErrors(cudaMemcpy(d_imageIn, h_imageIn, datasize, cudaMemcpyHostToDevice));

    // Setup Thread organization
    dim3 blockSize(16, 16);
    dim3 gridSize((height-1)/blockSize.x+1,(width-1)/blockSize.y+1);
    // PERFORM HISTOGRAM COMPUTATION
    compute_histograms<<<gridSize, blockSize>>>(d_imageIn, width, height, cpp, H_r, H_g, H_b);

    int numBlocks = 1;
    int numThreads = 3;
    cumulative_histograms<<<numBlocks, numThreads>>>(H_r, H_g, H_b, min_h);

    numBlocks = 3;
    numThreads = 256;
    new_intensities<<<numBlocks, numThreads>>>(H_r, H_g, H_b, newIntR, newIntG, newIntB, min_h, width, height, 256);

    assign_intensities<<<gridSize, blockSize>>>(d_imageIn, height, width, cpp, newIntR, newIntG, newIntB);

    checkCudaErrors(cudaMemcpy(h_imageIn, d_imageIn, datasize, cudaMemcpyDeviceToHost));
    getLastCudaError("copy_image() execution failed\n");
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    // Print time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel Execution time is: %0.3f milliseconds \n", milliseconds);

    // Write the output file
    char szImage_out_name_temp[255];
    strncpy(szImage_out_name_temp, szImage_out_name, 255);
    char *token = strtok(szImage_out_name_temp, ".");
    char *FileType = NULL;
    while (token != NULL)
    {
        FileType = token;
        token = strtok(NULL, ".");
    }

    if (!strcmp(FileType, "png"))
        stbi_write_png(szImage_out_name, width, height, cpp, h_imageIn, width * cpp);
    else if (!strcmp(FileType, "jpg"))
        stbi_write_jpg(szImage_out_name, width, height, cpp, h_imageIn, 100);
    else if (!strcmp(FileType, "bmp"))
        stbi_write_bmp(szImage_out_name, width, height, cpp, h_imageIn);
    else
        printf("Error: Unknown image format %s! Only png, bmp, or bmp supported.\n", FileType);

    // Free device memory
    checkCudaErrors(cudaFree(d_imageIn));
    checkCudaErrors(cudaFree(H_r));
    checkCudaErrors(cudaFree(H_g));
    checkCudaErrors(cudaFree(H_b));
    checkCudaErrors(cudaFree(newIntR));
    checkCudaErrors(cudaFree(newIntG));
    checkCudaErrors(cudaFree(newIntB));
    checkCudaErrors(cudaFree(min_h));

    // Clean-up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free host memory
    free(h_imageIn);
    // free(h_imageOut);

    return 0;
}
