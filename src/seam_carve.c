#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// Use 0 to retain the original number of color channels
#define COLOR_CHANNELS 0

void copy_image(unsigned char *image_out, const unsigned char *image_in, size_t size)
{

    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i)
    {
        image_out[i] = image_in[i];
    }
}

unsigned char* load_image(int *height, int *width, int *cpp, char image_title[255]) 
{
    // Load image from file and allocate space for the output image
    unsigned char *image = stbi_load(image_title, width, height, cpp, COLOR_CHANNELS);

    if (image == NULL)
    {
        printf("Error reading loading image %s!\n", image_title);
        exit(EXIT_FAILURE);
    }
    return image;
}

void save_image(unsigned char *image_out, int height, int width, int cpp, char image_out_title[255]) {
        // Write the output image to file
    char image_out_title_temp[255];
    strncpy(image_out_title_temp, image_out_title, 255);
    char *token = strtok(image_out_title_temp, ".");
    char *file_type = NULL;
    while (token != NULL)
    {
        file_type = token;
        token = strtok(NULL, ".");
    }

    if (!strcmp(file_type, "png"))
        stbi_write_png(image_out_title, width, height, cpp, image_out, width * cpp);
    else if (!strcmp(file_type, "jpg"))
        stbi_write_jpg(image_out_title, width, height, cpp, image_out, 100);
    else if (!strcmp(file_type, "bmp"))
        stbi_write_bmp(image_out_title, width, height, cpp, image_out);
    else
        printf("Error: Unknown image format %s! Only png, bmp, or bmp supported.\n", file_type);
}

void compute_energy(unsigned char *image, float *energy_image, int height, int width, int cpp)
{
    // pragma openmc for parallel
    for (int c = 0; c < cpp; c++) {
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                int partial_index = (c * height * width);
                //  G_x = -image[i-1, j-1] - 2*image[i, j-1] - image[i+1, j-1]
                //        +image[i-1, j+1] + 2*image[i, j+1] + image[i+1, j+1]
                //  G_y = -image[i-1, j-1] - 2*image[i, j-1] - image[i+1, j-1]
                //        +image[i-1, j+1] + 2*image[i, j+1] + image[i+1, j+1]
                int g_x;
                int g_y;
                // we have 8 edge cases (the pixel is in one of the corners or one of the border pixels)
                if (row == 0 && col == 0) {
                    // top left corner
                    g_x = -3*image[partial_index + row * width + col] - image[partial_index + row * width + (col + 1)]
                          +3*image[partial_index + row * width + col + 1] + image[partial_index + (row + 1) * width + (col + 1)];
                    
                    g_y = -3*image[partial_index + row * width + col] - image[partial_index + row * width + (col + 1)]
                          +3*image[partial_index + (row + 1) * width + col] + image[partial_index + (row + 1) * width + (col + 1)];
                } else if (row == 0 && col == width - 1) {
                    // top right corner
                    g_x = -3*image[partial_index + row * width + col - 1] - image[partial_index + (row + 1) * width + col - 1]
                          +3*image[partial_index + row * width + col] + image[partial_index + (row + 1) * width + col];
                    
                    g_y = -image[partial_index + row * width + col - 1] - 3*image[partial_index + row * width + col]
                          +image[partial_index + (row + 1) * width + col - 1] + 3*image[partial_index + (row + 1) * width + col];
                } else if (row == height - 1 && col == 0) {
                    // bottom left corner
                    g_x = -image[partial_index + (row - 1) * width + col] - 3*image[partial_index + row * width + col]
                          +image[partial_index + (row - 1) * width + col + 1] + 3*image[partial_index + row * width + col + 1];
                    
                    g_y = -3*image[partial_index + (row - 1) * width + col] - image[partial_index + (row - 1) * width + col + 1]
                          +3*image[partial_index + row * width + col] + image[partial_index + row * width + col + 1];
                } else if (row == height - 1 && col == width - 1) {
                    // bottom right corner
                    g_x = -image[partial_index + (row - 1) * width + col - 1] - 3*image[partial_index + row * width + col - 1]
                          +image[partial_index + (row - 1) * width + col] + 3*image[partial_index + row * width + col];
                    
                    g_y = -image[partial_index + (row - 1) * width + col - 1] - 3*image[partial_index + (row - 1) * width + col]
                          +image[partial_index + row * width + col - 1] + 3*image[partial_index + row * width + col];
                } else if (row == 0) {
                    // upper edge
                    g_x = -3*image[partial_index + row * width + col - 1] - image[partial_index + (row + 1) * width + col - 1]
                          +3*image[partial_index + row * width + col + 1] + image[partial_index + (row + 1) * width + col + 1];
                    
                    g_y = -image[partial_index + row * width + col - 1] - 2*image[partial_index + row * width + col] - image[partial_index + row * width + col + 1]
                          +image[partial_index + (row + 1) * width + col - 1] + 2*image[partial_index + (row + 1) * width + col] + image[partial_index + (row + 1) * width + col + 1];
                } else if (row == height - 1) {
                    // bottom edge
                    g_x = -image[partial_index + (row - 1) * width + col - 1] - 3*image[partial_index + row * width + col - 1]
                          +image[partial_index + (row - 1) * width + col + 1] + 3*image[partial_index + row * width + col + 1];
                    
                    g_y = -image[partial_index + (row - 1) * width + col - 1] - 2*image[partial_index + (row - 1) * width + col] - image[partial_index + (row - 1) * width + col + 1]
                          +image[partial_index + row * width + col - 1] + 2*image[partial_index + row * width + col] + image[partial_index + row * width + col + 1];
                } else if (col == 0) {
                    // left edge
                    g_x = -image[partial_index + (row - 1) * width + col] - 2*image[partial_index + row * width + col] - image[partial_index + (row + 1) * width + col]
                          +image[partial_index + (row - 1) * width + col + 1] + 2*image[partial_index + row * width + col + 1] + image[partial_index + (row + 1) * width + col + 1];
                    
                    g_y = -3*image[partial_index + (row - 1) * width + col] - image[partial_index + (row - 1) * width + col + 1]
                          +3*image[partial_index + (row + 1) * width + col] + image[partial_index + (row + 1) * width + col + 1];
                } else if (col == width - 1) {
                    // right edge
                    g_x = -image[partial_index + (row - 1) * width + col - 1] - 2*image[partial_index + row * width + col - 1] - image[partial_index + (row + 1) * width + col - 1]
                          +image[partial_index + (row - 1) * width + col] + 2*image[partial_index + row * width + col] + image[partial_index + (row + 1) * width + col];
                    
                    g_y = -image[partial_index + (row - 1) * width + col - 1] - 3*image[partial_index + (row - 1) * width + col]
                          +image[partial_index + (row + 1) * width + col - 1] + 3*image[partial_index + (row + 1) * width + col];
                } else {
                    // all other pixels
                    g_x = -image[partial_index + (row - 1) * width + col - 1] - 2*image[partial_index + row * width + col - 1] - image[partial_index + (row + 1) * width + col - 1]
                          +image[partial_index + (row - 1) * width + col + 1] + 2*image[partial_index + row * width + col + 1] + image[partial_index + (row + 1) * width + col + 1];
                    
                    g_y = -image[partial_index + (row - 1) * width + col - 1] - 2*image[partial_index + (row - 1) * width + col] - image[partial_index + (row - 1) * width + col + 1]
                          +image[partial_index + (row + 1) * width + col - 1] + 2*image[partial_index + (row + 1) * width + col] + image[partial_index + (row + 1) * width + col + 1];
                }

                // compute the average
                // odstrani deljenje s cpp, če bo prihajalo do prevelike računske napake
                int pixel_index = row * width + col;
                if (cpp == 0) {
                    energy_image[pixel_index] = sqrt((g_x*g_x) + (g_y*g_y)) / cpp;
                } else {
                    energy_image[pixel_index] += sqrt((g_x*g_x) + (g_y*g_y)) / cpp;
                }
            }
        }
    }
    /*
    // funkcija se uporabi za izračun povprečja, če v prejšnjem koraku ne delimo s cpp
    # pragma openmp ...
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            energy_image[i*height + j] = energy_image[i*height + j] / cpp;
        }
    }
    */
}

void identify_seam(float *energy_image, int *path, int height, int width, int cpp)
{
    // leave the bottom most row alone
    for (int row = height - 2; row >= 0; row--) {
        // inner loop can be completely parallel
        for (int col = 0; col < width; col++) {
            float min = energy_image[(row+1) * width + col];
            if (col != 0 && energy_image[(row+1) * width + col - 1] < min) {
                min = energy_image[(row+1) * width + col - 1];
            } else if (col != width - 1 && energy_image[(row+1) * width + col + 1] < min) {
                min = energy_image[(row+1) * width + col + 1];
            }
            energy_image[row * width + col] += min; 
        }
        // openm barrier
    }
}

int main(int argc, char *argv[]) {

    if (argc < 3)
    {
        printf("USAGE: sample input_image output_image\n");
        exit(EXIT_FAILURE);
    }

    char image_in_title[255];
    char image_out_title[255];

    snprintf(image_in_title, 255, "%s", argv[1]);
    snprintf(image_out_title, 255, "%s", argv[2]);

    int height;
    int width;
    int cpp;
    unsigned char *image_in = load_image(&height, &width, &cpp, image_in_title);

    const size_t datasize = width * height * cpp * sizeof(unsigned char);
    const size_t size_energy_img = width * height * sizeof(float);
    float *energy_image = (float *) malloc(size_energy_img);
    compute_energy(image_in, energy_image, height, width, cpp);

    const size_t path_length = height * sizeof(int);
    int *path = (int *) malloc(path_length);
    identify_seam(energy_image, path, height, width, cpp);

    unsigned char *image_out = (unsigned char *) malloc(datasize);

    save_image(image_out, height, width, cpp, image_out_title);
    // Release the memory
    free(image_in);
    free(image_out);

    return 0;
}