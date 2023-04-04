#include <stdio.h>
#include "../lodepng.h"
#include <CL/cl.h>


void readImage(char *filename, unsigned char **imageOut, unsigned *w, unsigned *h) {
    int err;
    err = lodepng_decode32_file(imageOut, w, h, filename);
    if (err) {
        printf("Error: Error when reading the image.");
    }
}

int readTextFile(char *filename, char *content, unsigned max_size) {
    FILE *file;
    file = fopen(filename, "r");
    char c;
    if (file == NULL) {
        return 1;
    }
    else {
        int i = 0;
        while ((c = fgetc(file)) != EOF && i < max_size-1) {
            content[i] = c;
            i++;
        }
        content[i] = '\0';
    }
    fclose(file);
    return 0;
}

void writeImage(char *filename, unsigned char *image, unsigned w, unsigned h) {
    int err;
    unsigned bitdepth = 8;
    LodePNGColorType colortype = LCT_GREY;

    err = lodepng_encode_file(filename, image, w, h, colortype, bitdepth);
    if (err) {
        printf("Error: Error when saving the image.");
    }
}


int moveToGPU(unsigned char *imageIn, cl_mem *imageOut, unsigned w, unsigned h, 
    cl_context context, cl_command_queue commands
) {
    cl_image_format imageFormat;
    int err;

    // Set image format
    imageFormat.image_channel_order = CL_RGBA;
    imageFormat.image_channel_data_type = CL_UNSIGNED_INT8;

    // Allocate memory on GPU
    *imageOut = clCreateImage2D(context, CL_MEM_READ_WRITE, &imageFormat, w, h, 0, NULL, &err);

    // Move images to GPU
    size_t origin[] = {0, 0, 0};
    size_t region[] = {w, h, 1};
    err  = clEnqueueWriteImage(commands, *imageOut, CL_TRUE, origin, region, 0, 0, imageIn, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to write image to GPU memory!\n");
        return 1;
    }
    return 0;

}
