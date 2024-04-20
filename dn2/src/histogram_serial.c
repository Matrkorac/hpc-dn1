#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <sys/time.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// Use 0 to retain the original number of color channels
#define COLOR_CHANNELS 0    


// Function to get the current time
double get_wall_time() {
    struct timeval time;
    if (gettimeofday(&time, NULL)) {
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}


int get_min(int cdf[255]) {
    int min = 0;
    for (int i = 0; i < 256 && min == 0; i++) {
        if (cdf[i] != 0 && cdf[i] < min)
            min = cdf[i];
    }
    return min;
}


void copy_image(unsigned char *image_out, const unsigned char *image_in, size_t size)
{

    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i)
    {
        image_out[i] = image_in[i];
    }
}


unsigned char* load_image(int *height, int *width, int *cpp, const char image_title[255]) 
{
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


/**
 * An image histogram represents the intensity distribution of a coulour in the digital image
 * --> image of size N x M
 * l_r, l_g, l_b in [0, L-1] (L=256)
 * n_l_c -> number of occurences of color c with intensity l
 * Three array (one for each channgel) H_r, H_g, H_b = [n_0_g, ... , n_255_g]
 * cummulative histogram (number of pixels of intensity of l or lower)
 * Equalization algorithm
 *  -> compute histograms H_r, H_g, H_b
 *  -> compute cumulative histogram for each color
 *  -> calculate new pixel intensities with the following formula:
 *      l_new_c = lower_bound((H_cdf_c - min(H_cdf_c)) / (N * M - min(H_cdf_c)) * (L - 1)))
 *  -> assign new intensity l_c_new to each colour channel of each pixel, corresponding to the original intensity l
 */
void compute_histograms(unsigned char *image, int height, int width, int cpp, int H_r[256], int H_g[256], int H_b[256]) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            H_r[image[(row * width + col) * cpp + 0]] += 1;
            H_g[image[(row * width + col) * cpp + 1]] += 1;
            H_b[image[(row * width + col) * cpp + 2]] += 1;
        }
    }
}


void cumulative_histogram(int H_in[256], int H_out[256]) {
    H_out[0] = H_in[0];
    for (int i = 1; i < 256; i++)
        H_out[i] = H_out[i-1] + H_in[i];
}


void new_intensities_for_c(int H_cdf[256], unsigned char new_intensities[256], int N, int M, int L) {
    /**
     *  calculate new pixel intensities with the following formula:
     *      l_new_c = lower_bound((H_cdf_c(l) - min(H_cdf_c)) / (N * M - min(H_cdf_c)) * (L - 1)))
     */
    int min_h = get_min(H_cdf);

    for (int i = 0; i < 256; i++)
        new_intensities[i] = (unsigned char) (((H_cdf[i] - min_h) / (N * M * min_h)) * (L - 1));
}


void reassing_intensities(unsigned char *image,
                        unsigned char new_intensities_r[256],
                        unsigned char new_intensities_g[256],
                        unsigned char new_intensities_b[256],
                        int width, int height, int cpp) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            image[(row * width + col) * cpp + 0] = new_intensities_r[image[(row * width + col) * cpp + 0]];
            image[(row * width + col) * cpp + 1] = new_intensities_g[image[(row * width + col) * cpp + 1]];
            image[(row * width + col) * cpp + 2] = new_intensities_b[image[(row * width + col) * cpp + 2]];
        }
    }
}


int main(int argc, char *argv[]) {
    // Check if the correct number of command-line arguments is provided
    if (argc < 3) {
        printf("USAGE: %s input_image output_image\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    // Extract command-line arguments
    const char *image_in_title = argv[1];
    const char *image_out_title = argv[2];
    int iterations = atoi(argv[3]);

    // Load the input image
    int height, width, cpp;

    double total_time = 0.0;
    unsigned char *image_in = load_image(&height, &width, &cpp, image_in_title);

    // Start timing
    double start_time = get_wall_time();

    int *H_r = (int *) calloc(256, sizeof(int));
    int *H_g = (int *) calloc(256, sizeof(int));
    int *H_b = (int *) calloc(256, sizeof(int));

    compute_histograms(image_in, height, width, cpp, H_r, H_g, H_b);

    int *H_rc = (int *) calloc(256, sizeof(int));
    int *H_gc = (int *) calloc(256, sizeof(int));
    int *H_bc = (int *) calloc(256, sizeof(int));

    cumulative_histogram(H_r, H_rc);
    cumulative_histogram(H_g, H_bc);
    cumulative_histogram(H_g, H_bc);

    free(H_r);
    free(H_g);
    free(H_b);

    unsigned char *new_intensities_r = (unsigned char *) calloc(256, sizeof(unsigned char));
    unsigned char *new_intensities_g = (unsigned char *) calloc(256, sizeof(unsigned char));
    unsigned char *new_intensities_b = (unsigned char *) calloc(256, sizeof(unsigned char));

    new_intensities_for_c(H_rc, new_intensities_r, height, width, 256);
    free(H_rc);
    new_intensities_for_c(H_gc, new_intensities_g, height, width, 256);
    free(H_gc);
    new_intensities_for_c(H_bc, new_intensities_b, height, width, 256);
    free(H_bc);

    reassing_intensities(image_in, new_intensities_r, new_intensities_g, new_intensities_b, width, height, cpp);

    free(new_intensities_r);
    free(new_intensities_g);
    free(new_intensities_b);

    // End timing
    double end_time = get_wall_time();

    // Calculate elapsed time
    total_time = (end_time - start_time);

    // Free memory for energy image and path
    free(image_in);

    printf("Time to compute: %f seconds\n", total_time);

    return 0;
}