#include <stdio.h>
#include <stdlib.h>
#include "lodepng.h"
#include <math.h>
#include <time.h>
#include <omp.h>
#include <CL/cl.h>

// TODO: Use OpenCL images type instead standart buffer.
// Refactor code


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

int downscaleImage(cl_mem imageIn, cl_mem *imageOut, unsigned w, unsigned h, 
    unsigned subpixels, unsigned factor, 
    cl_context context, cl_device_id device_id, cl_command_queue commands
) {
    int err = 0;
    size_t global = 1024;               // global domain size for our calculation
    size_t local = 64;                  // local domain size for our calculation
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
    cl_image_format imageFormat;        // image format

    // Set output image format
    imageFormat.image_channel_order = CL_RGBA;
    imageFormat.image_channel_data_type = CL_UNSIGNED_INT8;

    // Create image buffer in device memory
    *imageOut = clCreateImage2D(context, CL_MEM_READ_WRITE, &imageFormat, w/factor, h/factor, 0, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to create image on device!\n");
        return 1;
    }

    // Read the kernel code from file
    char *programSource;
    programSource = (char *) malloc(sizeof(char) * 2048);
    err = readTextFile("downscale.cl", programSource, 2048);
    if (err) {
        printf("Error: Error when reading the file.");
        return 1;
    }

    // Create and build the program
    program = clCreateProgramWithSource(context, 1, (const char **) &programSource, NULL, &err);
    if (!program) {
        printf("Error: Failed to create compute program!\n");
        return 1;
    }
    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    free(programSource);

    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return 1;
    }

    kernel = clCreateKernel(program, "downscaleImage", &err);
    if (!kernel || err != CL_SUCCESS) {
        printf("Error: Failed to create compute kernel!\n");
        return 1;
    }

    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &imageIn);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), imageOut);
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned), &w);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned), &h);
    err |= clSetKernelArg(kernel, 4, sizeof(unsigned), &factor);
    err |= clSetKernelArg(kernel, 5, sizeof(unsigned), &subpixels);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        return 1;
    }
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err) {
        printf("Error: Failed to execute kernel!\n");
        return 1;
    }
    clFinish(commands);

    clFlush(commands);

    clReleaseProgram(program);
    clReleaseKernel(kernel);
    return 0;
}

int grayscaleImage(cl_mem imageIn, cl_mem *imageOut,
    unsigned w, unsigned h, unsigned subpixels,
    cl_context context, cl_device_id device_id, cl_command_queue commands
) {
    int err = 0;
    size_t global = 1024;               // global domain size for our calculation
    size_t local = 64;                  // local domain size for our calculation
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
    cl_image_format imageFormat;

    // Set output image format
    imageFormat.image_channel_order = CL_R;
    imageFormat.image_channel_data_type = CL_UNSIGNED_INT8;

    // Create image buffer in device memory
    *imageOut = clCreateImage2D(context, CL_MEM_READ_WRITE, &imageFormat, w, h, 0, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to create image on device!\n");
        return 1;
    }

    char *programSource;
    programSource = (char *) malloc(sizeof(char) * 2048);
    err = readTextFile("grayscale.cl", programSource, 2048);
    if (err) {
        printf("Error: Error when reading the file.");
        return 1;
    }

    program = clCreateProgramWithSource(context, 1, (const char **) &programSource, NULL, &err);
    if (!program) {
        printf("Error: Failed to create compute program!\n");
        return 1;
    }
    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    free(programSource);

    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return 1;
    }

    kernel = clCreateKernel(program, "grayscaleImage", &err);
    if (!kernel || err != CL_SUCCESS) {
        printf("Error: Failed to create compute kernel!\n");
        return 1;
    }

    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &imageIn);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), imageOut);
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned), &w);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned), &h);
    err |= clSetKernelArg(kernel, 4, sizeof(unsigned), &subpixels);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        return 1;
    }
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err) {
        printf("Error: Failed to execute kernel!\n");
        return 1;
    }
    clFinish(commands);

    clFlush(commands);

    clReleaseProgram(program);
    clReleaseKernel(kernel);
    return 0;
}

void calcZNCC(unsigned char *imageL, unsigned char *imageR, unsigned char **imageOut, unsigned w, unsigned h, unsigned max_disp, unsigned win_size, int inv) {
    *imageOut = (unsigned char *) malloc(sizeof(unsigned char) * w * h);

    #pragma omp parallel for
    for (int j=0; j<h; j++) {
        double imgAvgL, imgAvgR;
        double zncc1, zncc2, zncc3, zncc;
        double znccBest, bestD;
        int x, y;
        unsigned countL, countR;

        for (int i=0; i<w; i++) {
            znccBest = -1;
            for (int d=0; d<=max_disp; d++) {
                // Calculate means over window
                imgAvgL = 0;
                imgAvgR = 0;
                countL = 0;
                countR = 0;
                for (int win_y=0; win_y<win_size; win_y++) {
                    y = j + win_y - win_size/2;
                    // Do not go outside the image
                    if ( y >= 0 && y < h ) {
                        for (int win_x=0; win_x<win_size; win_x++) {
                            x = i + win_x - win_size/2;
                            // Do not go outside the image
                            if ( x >= 0 && x < w ) {
                                // Add pixel values
                                imgAvgL += imageL[y*w + x];
                                countL++;
                            }
                            if ( x-d*inv >= 0 && x-d*inv < w ) {
                                imgAvgR += imageR[y*w + (x-d*inv)];
                                countR++;
                            }
                        }
                    }
                }
                // Calculate mean
                imgAvgL = imgAvgL / countL;
                imgAvgR = imgAvgR / countR;

                // Calculate ZNCC
                zncc1 = 0;
                zncc2 = 0;
                zncc3 = 0;
                for (int win_y=0; win_y<win_size; win_y++) {
                    y = j + win_y - win_size/2;
                    // Do not go outside the image
                    if ( y >= 0 && y < h ) {
                        for (int win_x=0; win_x<win_size; win_x++) {
                            x = i + win_x - win_size/2;
                            // Do not go outside the image
                            if ( x >= 0 && x-(d*inv) >= 0 && x < w && x-(d*inv) < w ) {
                                zncc1 += ((double)imageL[y*w + x] - imgAvgL) * ((double)imageR[y*w + x-(d*inv)] - imgAvgR);
                                zncc2 += pow((double)imageL[y*w + x] - imgAvgL, 2);
                                zncc3 += pow((double)imageR[y*w + x-(d*inv)] - imgAvgR, 2);
                            }
                        }
                    }
                }
                zncc = zncc1 / (sqrt(zncc2) * sqrt(zncc3));

                // Select if better than current best
                if (zncc > znccBest) {
                    znccBest = zncc;
                    bestD = d;
                }
            }
            (*imageOut)[j*w + i] = bestD;
        }
    }
}


void normalizeImage(unsigned char *imageIn, unsigned char **imageOut, unsigned w, unsigned h) {
    unsigned min, max;
    min = 255;
    max = 0;

    for (int i=0; i<w*h; i++) {
        if (imageIn[i] < min) {
            min = imageIn[i];
        }
        if (imageIn[i] > max) {
            max = imageIn[i];
        }
    }
    *imageOut = (unsigned char *) malloc(sizeof(unsigned char) * w * h);
    #pragma omp parallel for
    for (int i=0; i<w*h; i++) {
        (*imageOut)[i] = (imageIn[i] - min) * 255 / (max - min);
    }
}


void crossCheck(unsigned char *image1, unsigned char *image2, unsigned char **imageOut, unsigned w, unsigned h, unsigned threshold) {
    *imageOut = (unsigned char *) malloc(sizeof(unsigned char) * w * h);

    #pragma omp parallel for
    for (int i=0; i<h; i++) {
        int x, y, d;
        for (int j=0; j<w; j++) {
            y = i * w;
            x = j;
            d = image1[y + x];

            if ( y + x - d >= 0 && abs((char)image1[y + x] - (char)image2[y + (x-d)]) > threshold) {
                (*imageOut)[y + x] = 0;
            }
            else {
                (*imageOut)[y + x] = image1[y + x];
            }
        }
    }
}

void occlusionFill(unsigned char *imageIn, unsigned char **imageOut, unsigned w, unsigned h) {
    const int MAX_DIST = 1000;
    *imageOut = (unsigned char *) malloc(sizeof(unsigned char) * w * h);
    #pragma omp parallel for
    for (int i=0; i<h; i++) {
        unsigned stop;
        for (int j=0; j<w; j++) {
            if (imageIn[i*w + j] == 0) {
                // Find the closest
                stop = 0;
                for (int d=1; d<=MAX_DIST && !stop; d++) {
                    for (int x=-d+1; x<=d-1 && !stop; x++) {
                        if (i+x>=0 && i+x<h) {
                            if (j-d>=0 && imageIn[(i+x)*w + j-d] != 0) {
                                (*imageOut)[i*w + j] = imageIn[(i+x)*w + j-d];
                                stop = 1;
                            }
                            else if (j+d<w && imageIn[(i+x)*w + j + d] != 0) {
                                (*imageOut)[i*w + j] = imageIn[(i+x)*w + j+d];
                                stop = 1;
                            }
                        }
                    }
                    for (int y=-d; y<=d && !stop; y++) {
                        if (j+y>=0 &&j+y<w) {
                            if (i-d>=0 && imageIn[(i-d)*w + j+y] != 0) {
                                (*imageOut)[i*w + j] = imageIn[(i-d)*w + j+y];
                                stop = 1;
                            }
                            else if (i+d<h && imageIn[(i+d)*w + j+y] != 0 ) {
                                (*imageOut)[i*w + j] = imageIn[(i+d)*w + j+y];
                                stop = 1;
                            }
                        }
                    }
                }
            }
            else {
                (*imageOut)[i*w + j] = imageIn[i*w + j];
            }
        }
    }
}



int main(void) {

    // Variables
    unsigned const MAX_DISP = 260/4;
    unsigned const WIN_SIZE = 9;
    unsigned const THRESHOLD = 8;
    char file1[] = "dataset\\im0.png";
    char file2[] = "dataset\\im1.png";
    char file3[] = "dataset\\depthmap.png";
    unsigned char *image1, *image2, *imageOut;
    unsigned w, h;
    unsigned subpixels = 4; 
    unsigned scaleFactor = 4;
    clock_t start, end;
    double timeElapsed;
    int err;


    // Read images
    // start = clock();
    #pragma omp parallel sections
    {
        #pragma omp section 
        readImage(file1, &image1, &w, &h);
        #pragma omp section
        readImage(file2, &image2, &w, &h);
    }
    // end = clock();
    // timeElapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
    // printf("Read files: %.3f s\n", timeElapsed);

    // OpenCL variables
    cl_uint num_platforms;
    cl_uint num_devices;

    cl_platform_id *platform_id;
    cl_device_id device_id;             // compute device id 
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue

    cl_mem image1GPU, image2GPU, imageDs1GPU, imageDs2GPU, imageGrayGPU1, imageGrayGPU2, imageOutGPU;

    // Prepare OpenCL
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to get the number of platforms!\n");
        return 1;
    }
    platform_id = (cl_platform_id *) malloc(sizeof(cl_platform_id) * num_platforms);
    err = clGetPlatformIDs(num_platforms, platform_id, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to get platform id!\n");
        return 1;
    }
    err = clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);
        if (err != CL_SUCCESS) {
        printf("Error: Failed to create a device group!\n");
        return 1;
    }
    context = clCreateContext(0, num_devices, &device_id, NULL, NULL, &err);
    if (!context) {
        printf("Error: Failed to create a compute context!\n");
        return 1;
    }
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands) {
        printf("Error: Failed to create a command commands!\n");
        return 1;
    }
 
    cl_image_format imageFormat;
    imageFormat.image_channel_order = CL_RGBA;
    imageFormat.image_channel_data_type = CL_UNSIGNED_INT8;
    // Allocate memory on GPU
    image1GPU = clCreateImage2D(context, CL_MEM_READ_WRITE, &imageFormat, w, h, 0, NULL, &err);
    image2GPU = clCreateImage2D(context, CL_MEM_READ_WRITE, &imageFormat, w, h, 0, NULL, &err);
    // imageDs1GPU = clCreateImage2D(context, CL_MEM_READ_WRITE, &imageFormat, w/scaleFactor, h/scaleFactor, 0, NULL, &err);
    // imageDs2GPU = clCreateImage2D(context, CL_MEM_READ_WRITE, &imageFormat, w/scaleFactor, h/scaleFactor, 0, NULL, &err);
    // imageOutGPU = clCreateImage2D(context, CL_MEM_WRITE_ONLY, &imageFormat, w/scaleFactor, h/scaleFactor, 0, NULL, &err);

    // image2GPU = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(unsigned char) * h * w * subpixels, NULL, NULL);
    // imageDs1GPU = clCreateBuffer(context,  CL_MEM_READ_WRITE,  sizeof(unsigned char) * h/scaleFactor * w/scaleFactor, NULL, NULL);
    // imageDs2GPU = clCreateBuffer(context,  CL_MEM_READ_WRITE,  sizeof(unsigned char) * h/scaleFactor * w/scaleFactor, NULL, NULL);
    // imageOutGPU = clCreateBuffer(context,  CL_MEM_WRITE_ONLY,  sizeof(unsigned char) * h/scaleFactor * w/scaleFactor * 4 , NULL, NULL);

    // Move images to GPU
    size_t origin[] = {0, 0, 0};
    size_t region[] = {w, h, 1};
    err  = clEnqueueWriteImage(commands, image1GPU, CL_TRUE, origin, region, 0, 0, image1, 0, NULL, NULL);
    err |= clEnqueueWriteImage(commands, image2GPU, CL_TRUE, origin, region, 0, 0, image2, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to move images to the device!\n");
        return 1;
    }


    // Downscale by four
    start = clock();
    err  = downscaleImage(image1GPU, &imageDs1GPU, w, h, subpixels, scaleFactor, context, device_id, commands);
    err |= downscaleImage(image2GPU, &imageDs2GPU, w, h, subpixels, scaleFactor, context, device_id, commands);
    if (err) {
        printf("Error: Failed to downscale image!\n");
        return 1;
    }
    w = w / scaleFactor;
    h = h / scaleFactor;
    end = clock();
    timeElapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Downscale: %.3f s\n", timeElapsed);

    // Convert to grayscale
    start = clock();
    err  = grayscaleImage(imageDs1GPU, &imageGrayGPU1, w, h, subpixels, context, device_id, commands);
    err |= grayscaleImage(imageDs2GPU, &imageGrayGPU2, w, h, subpixels,  context, device_id, commands);
    if (err) {
        printf("Error: Failed to convert to grayscale!\n");
        return 1;
    }
    end = clock();
    timeElapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Grayscale: %.3f s\n", timeElapsed);

    // Do ZNCC
    // start = clock();


    // end = clock();
    // timeElapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
    // printf("ZNCC: %.3f s\n", timeElapsed);

    // Cross checking
    // start = clock();

    // end = clock();
    // timeElapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
    // printf("Cross check: %.3f s\n", timeElapsed);

    // Occlusion fill
    // start = clock();

    // end = clock();
    // timeElapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
    // printf("Occlusion fill: %.3f s\n", timeElapsed);

    // Normalize image
    // start = clock();

    // end = clock();
    // timeElapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
    // printf("Normalization: %.3f s\n", timeElapsed);


    region[0] = w;
    region[1] = h;
    region[2] = 1;
    imageOut = (unsigned char *) malloc(sizeof(unsigned char) * h * w);
    clEnqueueReadImage(commands, imageGrayGPU1, CL_TRUE, origin, region, 0, 0, imageOut, 0, NULL, NULL);
    // clEnqueueReadBuffer(commands, imageGrayGPU1, CL_TRUE, 0, sizeof(unsigned char) * h * w * 4, imageOut, 0, NULL, NULL );

    // Save image
    start = clock();
    writeImage(file3, imageOut, w, h);
    end = clock();
    timeElapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Save file: %.3f s\n", timeElapsed);

    // Free memory
    free(image1);
    free(image2);
    free(imageOut);

    clReleaseMemObject(image1GPU);
    clReleaseMemObject(image2GPU);
    clReleaseMemObject(imageDs1GPU);
    // clReleaseMemObject(imageDs2GPU);
    clReleaseMemObject(imageGrayGPU1);
    // clReleaseMemObject(imageGrayGPU2);

    // clReleaseMemObject(imageOutGPU);

    return 0;
}





    // imageOut = (unsigned char *) malloc(sizeof(unsigned char) * h * w * 4);
    // clEnqueueReadBuffer(commands, imageOutGPU, CL_TRUE, 0, sizeof(unsigned char) * h * w * 4, imageOut, 0, NULL, NULL );