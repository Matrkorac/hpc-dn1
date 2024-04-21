#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
                                    int H[3*256])
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
    atomicAdd(&H[0 + threadIdx.x * blockDim.y + threadIdx.y], Hrs[threadIdx.x * blockDim.y + threadIdx.y]);
    atomicAdd(&H[256 + threadIdx.x * blockDim.y + threadIdx.y], Hgs[threadIdx.x * blockDim.y + threadIdx.y]);
    atomicAdd(&H[512 + threadIdx.x * blockDim.y + threadIdx.y], Hbs[threadIdx.x * blockDim.y + threadIdx.y]);
}


// naive implemetation with only three threads
__global__ void cumulative_histograms(int H[3*256], int *min_h)
{
    int idx = threadIdx.x;

    if (idx == 0) {
        int min_r = H[0];
        for (int i = 1; i < 256; i++) {
            H[0 + i] = H[0 + i] + H[0 + i-1];
            if (min_r == 0 && H[i] != 0)
                min_r = H[i];
        }
        min_h[0] = min_r;
    }

    if (idx == 1) {
        int min_g = H[256];
        for (int i = 1; i < 256; i++) {
            H[256 + i] = H[256 + i] + H[256 + i-1];
            if (min_g == 0 && H[256+i] != 0) 
                min_g = H[256+i];
        }
        min_h[1] = min_g;
    }

    if (idx == 2) {
        int min_b = H[512];
        for (int i = 1; i < 256; i++) {
            H[512 + i] = H[512 + i] + H[512 + i-1];
            if (min_b == 0 && H[512+i] != 0) 
                min_b = H[512+i];
        }
        min_h[2] = min_b;
    }
}


// efficient parallel scan algorithm for cumulative histograms
__global__ void cumulative_histograms_scan(int H[3*256], int *min_h, int N)
{
    // thee blocks of 256 threads (16x16)
    int channelIdx = blockIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int step;
    int lim = (int) log2(N);
    int k2 = 1;
    int k2m1 = 2;
    int step = 2;

    // upsweep phase
    for (int k = 0; k <= lim - 1; ++k) {
        if (idx < N - 1 && idx % step == 0) {
            // for (int i = 0; i <= N-1; i += step) {
            H[channelIdx * 256 + idx-1+k2m1] = H[channelIdx * 256 + idx-1+k2] + H[channelIdx * 256 + idx-1+k2m1];
        }
        k2 *= 2;
        k2m1 *= 2;
        step *= 2;
        __syncthreads();
    }

    k2m1 /= 4; 
    // downsweep phase
    for (int k = lim; k >= 1; --k) {
        if (idx < N - 1 && idx % step == 0) {
            // for (int i = 0; i <= N - 1; i += step) {
            H[channelIdx * 256 + idx-1+k2m1+2^k] = H[channelIdx * 256 + idx-1+k2m1+k2] + H[channelIdx * 256 + idx-1+k2];
        }
        k2 /= 2;
        k2m1 /= 2;
        step /= 2;
        __syncthreads();
    }
}


// can be completely parallelized
__global__ void new_intensities(int H[3*256],
                                unsigned char new_int[3*256],
                                int* min_h,
                                int N, int M, int L)
{
    // can have three blocks; one for each color channel
    // one block of 256 threads (16x16)
    int idx = blockIdx.x * threadIdx.y * blockDim.x + threadIdx.x;

    new_int[blockIdx.x * 256 + idx] = (unsigned char) (((H[blockIdx.x * 256 + idx] - min_h[blockIdx.x]) / (N * M * min_h[blockIdx.x])) * (L - 1));
}


// can be completely parallelized
__global__ void assign_intensities(unsigned char *image, int height, int width, int cpp,
                                    unsigned char intensities[3 * 256])
{
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;

    for (int i = gidx; i < height; i += blockDim.x * gridDim.x)
    {
        for (int j = gidy; j < width; j += blockDim.y * gridDim.y)
        {
            image[(i * width + j) * cpp + 0] = intensities[image[(i * width + j) * cpp + 0]];
            image[(i * width + j) * cpp + 1] = intensities[256 + image[(i * width + j) * cpp + 1]];
            image[(i * width + j) * cpp + 2] = intensities[512 + image[(i * width + j) * cpp + 2]];
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

    int *H;
    const size_t histogram_size = 3 * 256 * sizeof(int);

    unsigned char *newIntensities;
    const size_t intensities_size = 3 * 256 * sizeof(unsigned char);

    int *min_h;
    const size_t min_h_size = 3 * sizeof(int);

    // Allocate memory on the device
    checkCudaErrors(cudaMalloc(&d_imageIn, datasize));
    // checkCudaErrors(cudaMalloc(&d_imageOut, datasize));

    // allocate the histograms for color channels to the device
    checkCudaErrors(cudaMalloc(&H, histogram_size));

    checkCudaErrors(cudaMalloc(&newIntensities, intensities_size));

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
    compute_histograms<<<gridSize, blockSize>>>(d_imageIn, width, height, cpp, H);

    int numBlocks = 1;
    int numThreads = 3;
    cumulative_histograms<<<numBlocks, numThreads>>>(H, min_h);

    numBlocks = 3;
    numThreads = 256;
    new_intensities<<<numBlocks, numThreads>>>(H, newIntensities, min_h, width, height, 256);

    assign_intensities<<<gridSize, blockSize>>>(d_imageIn, height, width, cpp, newIntensities);

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
