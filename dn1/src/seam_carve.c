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

void compute_energy(unsigned char *image, double *energy_image, int height, int width, int cpp)
{
    // index = (i * width + j) * channels + k                       
    // pragma openmc for parallel
	#pragma omp parallel for schedule(dynamic)
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            for (int c = 0; c < cpp; c++) {
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
                    g_x = -3*image[(row * width + col) * cpp + c] - image[(row * width + (col + 1)) * cpp + c]
                          +3*image[(row * width + col + 1) * cpp + c] + image[((row + 1) * width + (col + 1)) * cpp + c];
                    
                    g_y = -3*image[(row * width + col) * cpp + c] - image[(row * width + (col + 1)) * cpp + c]
                          +3*image[((row + 1) * width + col) * cpp + c] + image[((row + 1) * width + (col + 1)) * cpp + c];
                } else if (row == 0 && col == width - 1) {
                    // top right corner
                    g_x = -3*image[(row * width + col - 1) * cpp + c] - image[((row + 1) * width + col - 1) * cpp + c]
                          +3*image[(row * width + col) * cpp + c] + image[((row + 1) * width + col) * cpp + c];
                    
                    g_y = -image[(row * width + col - 1) * cpp + c] - 3*image[(row * width + col) * cpp + c]
                          +image[((row + 1) * width + col - 1) * cpp + c] + 3*image[((row + 1) * width + col) * cpp + c];
                } else if (row == height - 1 && col == 0) {
                    // bottom left corner
                    g_x = -image[((row - 1) * width + col) * cpp + c] - 3*image[(row * width + col) * cpp + c]
                          +image[((row - 1) * width + col + 1) * cpp + c] + 3*image[(row * width + col + 1) * cpp + c];
                    
                    g_y = -3*image[((row - 1) * width + col) * cpp + c] - image[((row - 1) * width + col + 1) * cpp + c]
                          +3*image[(row * width + col) * cpp + c] + image[(row * width + col + 1) * cpp + c];
                } else if (row == height - 1 && col == width - 1) {
                    // bottom right corner
                    g_x = -image[((row - 1) * width + col - 1) * cpp + c] - 3*image[(row * width + col - 1) * cpp + c]
                          +image[((row - 1) * width + col) * cpp + c] + 3*image[(row * width + col) * cpp + c];
                    
                    g_y = -image[((row - 1) * width + col - 1) * cpp + c] - 3*image[((row - 1) * width + col) * cpp + c]
                          +image[(row * width + col - 1) * cpp + c] + 3*image[(row * width + col) * cpp + c];
                } else if (row == 0) {
                    // upper edge
                    g_x = -3*image[(row * width + col - 1) * cpp + c] - image[((row + 1) * width + col - 1) * cpp + c]
                          +3*image[(row * width + col + 1) * cpp + c] + image[((row + 1) * width + col + 1) * cpp + c];
                    
                    g_y = -image[(row * width + col - 1) * cpp + c] - 2*image[(row * width + col) * cpp + c] - image[(row * width + col + 1) * cpp + c]
                          +image[((row + 1) * width + col - 1) * cpp + c] + 2*image[((row + 1) * width + col) * cpp + c] + image[((row + 1) * width + col + 1) * cpp + c];
                } else if (row == height - 1) {
                    // bottom edge
                    g_x = -image[((row - 1) * width + col - 1) * cpp + c] - 3*image[(row * width + col - 1) * cpp + c]
                          +image[((row - 1) * width + col + 1) * cpp + c] + 3*image[(row * width + col + 1) * cpp + c];
                    
                    g_y = -image[((row - 1) * width + col - 1) * cpp + c] - 2*image[((row - 1) * width + col) * cpp + c] - image[((row - 1) * width + col + 1) * cpp + c]
                          +image[(row * width + col - 1) * cpp + c] + 2*image[(row * width + col) * cpp + c] + image[(row * width + col + 1) * cpp + c];
                } else if (col == 0) {
                    // left edge
                    g_x = -image[((row - 1) * width + col) * cpp + c] - 2*image[(row * width + col) * cpp + c] - image[((row + 1) * width + col) * cpp + c]
                          +image[((row - 1) * width + col + 1) * cpp + c] + 2*image[(row * width + col + 1) * cpp + c] + image[((row + 1) * width + col + 1) * cpp + c];
                    
                    g_y = -3*image[((row - 1) * width + col) * cpp + c] - image[((row - 1) * width + col + 1) * cpp + c]
                          +3*image[((row + 1) * width + col) * cpp + c] + image[((row + 1) * width + col + 1) * cpp + c];
                } else if (col == width - 1) {
                    // right edge
                    g_x = -image[((row - 1) * width + col - 1) * cpp + c] - 2*image[(row * width + col - 1) * cpp + c] - image[((row + 1) * width + col - 1) * cpp + c]
                          +image[((row - 1) * width + col) * cpp + c] + 2*image[(row * width + col) * cpp + c] + image[((row + 1) * width + col) * cpp + c];
                    
                    g_y = -image[((row - 1) * width + col - 1) * cpp + c] - 3*image[((row - 1) * width + col) * cpp + c]
                          +image[((row + 1) * width + col - 1) * cpp + c] + 3*image[((row + 1) * width + col) * cpp + c];
                } else {
                    // all other pixels
                    g_x = -image[((row - 1) * width + col - 1) * cpp + c] - 2*image[(row * width + col - 1) * cpp + c] - image[((row + 1) * width + col - 1) * cpp + c]
                          +image[((row - 1) * width + col + 1) * cpp + c] + 2*image[(row * width + col + 1) * cpp + c] + image[((row + 1) * width + col + 1) * cpp + c];
                    
                    g_y = -image[((row - 1) * width + col - 1) * cpp + c] - 2*image[((row - 1) * width + col) * cpp + c] - image[((row - 1) * width + col + 1) * cpp + c]
                          +image[((row + 1) * width + col - 1) * cpp + c] + 2*image[((row + 1) * width + col) * cpp + c] + image[((row + 1) * width + col + 1) * cpp + c];
                }

                // compute the average
                // odstrani deljenje s cpp, če bo prihajalo do prevelike računske napake
                int pixel_index = row * width + col;
                if (cpp == 0) {
                    energy_image[pixel_index] = sqrt((g_x*g_x) + (g_y*g_y));
                } else {
                    energy_image[pixel_index] += sqrt((g_x*g_x) + (g_y*g_y));
                }
            }
        }
    }
    
    // funkcija se uporabi za izračun povprečja, če v prejšnjem koraku ne delimo s cpp
    // # pragma openmp ...
	#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            energy_image[i*width + j] /= cpp;
        }
    }
}

void cumulative_energy(double *energy_image, int height, int width)
{
    // leave the bottom most row alone
    for (int row = height - 2; row >= 0; row--) {
        // inner loop can be completely parallel
        #pragma omp parallel for schedule(static)
        for (int col = 0; col < width; col++) {
            double min = energy_image[(row+1) * width + col];
            if (col != 0 && energy_image[(row+1) * width + col - 1] < min) {
                min = energy_image[(row+1) * width + col - 1];
            }
            if (col != width - 1 && energy_image[(row+1) * width + col + 1] < min) {
                min = energy_image[(row+1) * width + col + 1];
            }
            energy_image[row * width + col] += min; 
        }
        #pragma omp barrier
        // openm barrier
    }
}


// funkcija za iskanje indexa z min vrednostjo v tabeli
int find_min_indx(double* energy, int width)
{
    int min_idx = 0;
    // TODO ne vem, če se da dobro paralizirat ker mora biti min_index globalni za vse niti
    for (int col = 1; col < width; col++) {
        // ker gledamo samo prvo vrstico, nam vrstice ni treba upoštevati
        if (energy[col] < energy[min_idx]) {
            min_idx = col;
        }
    }
    return min_idx;
}

void find_seam(double *energy_image, int *seam_index, int height, int width)
{
    // v prvi vstici
    seam_index[0] = find_min_indx(energy_image, width);
    for (int row = 1; row < height; row++) {
        int prev_idx = seam_index[row-1];
        int min_idx = prev_idx;
        // preverimo levi element
        if (prev_idx != 0 && energy_image[row * width + prev_idx - 1] < energy_image[row * width + prev_idx]) {
            min_idx--;
        }
        // preverimo desni element
        if (prev_idx != width - 1 && energy_image[row * width + prev_idx + 1] < energy_image[row * width + prev_idx]) {
            min_idx++;
        }
        // printf("%d ", min_idx);
        seam_index[row] = min_idx;
    }
    // printf("\n");
}

void remove_seam(unsigned char *image, unsigned char *img_reduced, int *path, int height, int width, int cpp) 
{
    // index = (i * width + j) * channels + k                       
    // tole zunanjo se da paralelizirat
	#pragma omp parallel for schedule(static)
    for (int row = 0; row < height; row++) {
        // notranje na žalost ne
        for (int col = 0; col < width; col++) {
            for (int c = 0; c < cpp; c++) {
                if (col < path[row]) {
                    img_reduced[(row * (width-1) + col) * cpp + c] = image[(row * width + col) * cpp + c];
                } else if (col > path[row]) {
                    img_reduced[(row * (width-1) + (col-1)) * cpp + c] = image[(row * width + col) * cpp + c];
                }
            }
        }
    }
}

// Function to get the current time
double get_wall_time() {
    struct timeval time;
    if (gettimeofday(&time, NULL)) {
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

int main(int argc, char *argv[]) {
    // Check if the correct number of command-line arguments is provided
    if (argc < 4) {
        printf("USAGE: %s input_image output_image num_seams_to_remove\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    // Extract command-line arguments
    const char *image_in_title = argv[1];
    const char *image_out_title = argv[2];
    int seams_to_remove = atoi(argv[3]);

    // Load the input image
    int height, orig_width, width, cpp;

    // Define different numbers of cores to test
    int num_cores[] = {1, 2, 4};

    // Define the number of runs to average time measurements
    int num_runs = 3;

    // Initialize variables to store optimal configuration
    double min_time = INFINITY;
    int optimal_cores = 0;


    // Perform time measurements for different numbers of cores
    for (int k = 0; k < sizeof(num_cores) / sizeof(num_cores[0]); k++) {
        int cores = num_cores[k];
        omp_set_num_threads(cores);

        double total_time = 0.0;
        for (int run = 0; run < num_runs; run++) {
            unsigned char *image_in = load_image(&height, &orig_width, &cpp, image_in_title);

            // Start timing
            double start_time = get_wall_time();
            width = orig_width;

            // Allocate memory for the output image
            // unsigned char *image_out = (unsigned char *)malloc(height * width * cpp * sizeof(unsigned char));

            // // Copy input image to output image
            // copy_image(image_out, image_in, height * width * cpp * sizeof(unsigned char));

            //  tole mora bit sekvenčno
            for (int i = 0; i < seams_to_remove; i++) {

                const size_t size_energy_img = width * height * sizeof(double);
                double *energy_image = (double *) malloc(size_energy_img);
                compute_energy(image_in, energy_image, height, width, cpp);

                cumulative_energy(energy_image, height, width);

                // Allocate memory for energy image and path outside the loop
                const size_t path_length = height * sizeof(int);
                int *path = (int *) malloc(path_length);
                find_seam(energy_image, path, height, width);
                
                free(energy_image);

                const size_t size_reduced = (width-1) * height * cpp * sizeof(unsigned char);
                unsigned char *img_reduced = (unsigned char *) malloc(size_reduced);
                remove_seam(image_in, img_reduced, path, height, width, cpp);
                free(image_in);
                image_in = img_reduced;
                free(img_reduced);
                width--; // Update width
                free(path);
            }

            // End timing
            double end_time = get_wall_time();

            // Calculate elapsed time
            total_time += (end_time - start_time);

            // Free memory for energy image and path
            if (run == num_runs - 1) {
                // Modify the output image path to include the number of cores
                char output_path[255];
                snprintf(output_path, sizeof(output_path), "%s_%d.png", image_out_title, cores);
                save_image(image_in, height, width, cpp, output_path);
                printf("%s\n", output_path);
            }
            free(image_in);
        }

        // Calculate average time for the current configuration
        double avg_time = total_time / num_runs;

        // Update optimal configuration if necessary
        if (avg_time < min_time) {
            min_time = avg_time;
            optimal_cores = cores;
        }
        printf("Average time with %d cores: %f seconds\n", cores, avg_time);
    }

    printf("Optimal number of cores: %d\n", optimal_cores);

    return 0;
}