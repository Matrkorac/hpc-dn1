#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

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
    unsigned char *image_out = (unsigned char *)malloc(datasize);
    
    save_image(image_out, height, width, cpp, image_out_title);
    // Release the memory
    free(image_in);
    free(image_out);

    return 0;
}