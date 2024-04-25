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
                                    int *H)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        atomicAdd(&H[image[(row * width + col) * cpp + 0]], 1);
        atomicAdd(&H[256 + image[(row * width + col) * cpp + 1]], 1);
        atomicAdd(&H[512 + image[(row * width + col) * cpp + 2]], 1);
    }
}

__global__ void cumulative_histograms(int *H, int *min_h)
{
    int idx = threadIdx.x;

    if (idx < 256) {
        int min_r = H[idx];
        int min_g = H[256 + idx];
        int min_b = H[512 + idx];
        for (int i = 1; i < 256; i++) {
            H[idx + i] += H[idx + i - 1];
            H[256 + idx + i] += H[256 + idx + i - 1];
            H[512 + idx + i] += H[512 + idx + i - 1];
            if (min_r == 0 && H[idx + i] != 0)
                min_r = H[idx + i];
            if (min_g == 0 && H[256 + idx + i] != 0) 
                min_g = H[256 + idx + i];
            if (min_b == 0 && H[512 + idx + i] != 0) 
                min_b = H[512 + idx + i];
        }
        min_h[0] = min_r;
        min_h[1] = min_g;
        min_h[2] = min_b;
    }
}

__global__ void cumulative_histograms_test(int H[3*256], int *min_h)
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
    // thee blocks of 256 threads
    int channelIdx = blockIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lim = (int) log2(N);
    int k2 = 1;
    int k2m1 = 2;
    int step = 2;

    // we have 3 * 256 threads, one thread for each index in the channel
    // each thread lives through the whole algorithm
    // a thread computes the final cumulative sum for each index of the given array

    // upsweep phase
    for (int k = 0; k <= lim - 1; ++k) {
        if (idx < N - 1 && idx % step == 0) {
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
            H[channelIdx * 256 + idx-1+k2m1+2^k] = H[channelIdx * 256 + idx-1+k2m1+k2] + H[channelIdx * 256 + idx-1+k2];
        }
        k2 /= 2;
        k2m1 /= 2;
        step /= 2;
        __syncthreads();
    }
}

__global__ void new_intensities(int H[3*256],
                                unsigned char new_int[3*256],
                                int* min_h,
                                int N, int M, int L)
{
    // can have three blocks; one for each color channel
    // one block of 256 threads
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    new_int[blockIdx.x * 256 + idx] = (unsigned char) (((H[idx] - min_h[blockIdx.x]) / (N * M - min_h[blockIdx.x])) * (L - 1));
}


__global__ void assign_intensities(unsigned char *image, int height, int width, int cpp,
                                    unsigned char intensities[3 * 256])
{
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;

    // each thread assigns new intensities for all three color channels
    // for thread assigns WIDTH / (BLOCKDIM.X * GRIDIM.X)
    // for gridSize(3, 3) and blockSize(16, 16) we get --> WIDTH / (3 * 16)
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

    unsigned char *d_imageIn;
    unsigned char *d_newIntensities;
    int *d_H;
    int *d_min_h;

    // allocate memory on the device
    checkCudaErrors(cudaMalloc(&d_imageIn, datasize));
    checkCudaErrors(cudaMalloc(&d_newIntensities, 3 * 256 * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc(&d_H, 3 * 256 * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_min_h, 3 * sizeof(int)));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_time = 0;

    for (int i = 0; i < 5; i++) {
        cudaEventRecord(start);
        checkCudaErrors(cudaMemcpy(d_imageIn, h_imageIn, datasize, cudaMemcpyHostToDevice));
        // compute histograms
        dim3 blockSize(16, 16);
        dim3 gridSize((height + blockSize.x - 1) / blockSize.x, (width + blockSize.y - 1) / blockSize.y);
        compute_histograms<<<gridSize, blockSize>>>(d_imageIn, width, height, cpp, d_H);

        // compute cumulative histograms
        cumulative_histograms_test<<<3, 256>>>(d_H, d_min_h);
        // cumulative_histograms_scan<<<1, 3>>>(d_H, d_min_h, 256);

        // compute new intensities
        new_intensities<<<3, 256>>>(d_H, d_newIntensities, d_min_h, width, height, 256);

        // assign new intensities to the image
        assign_intensities<<<gridSize, blockSize>>>(d_imageIn, height, width, cpp, d_newIntensities);

        checkCudaErrors(cudaMemcpy(h_imageIn, d_imageIn, datasize, cudaMemcpyDeviceToHost));

        cudaEventRecord(stop);

        cudaEventSynchronize(stop);

        // Print time
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_time += milliseconds;
    }
    total_time /= 5;
    total_time /= 1000;
    printf("Average Execution time is: %0.4f seconds \n", total_time);

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
    checkCudaErrors(cudaFree(d_newIntensities));
    checkCudaErrors(cudaFree(d_H));
    checkCudaErrors(cudaFree(d_min_h));

    // Free host memory
    stbi_image_free(h_imageIn);

    return 0;
}
