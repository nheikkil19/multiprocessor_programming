#include <stdio.h>
#include <stdlib.h>
#include "../lodepng.h"
#include <math.h>
#include <time.h>
#include <omp.h>
#include <CL/cl.h>
#include "myutils.c"
#include "transformations.c"
#include "zncc.c"

int main(void) {

    // Variables
    unsigned const MAX_DISP = 260/4;
    unsigned const WIN_SIZE = 9;
    unsigned const THRESHOLD = 8;
    char file1[] = "..\\dataset\\im0.png";
    char file2[] = "..\\dataset\\im1.png";
    char file3[] = "..\\dataset\\depthmap.png";
    unsigned char *image1, *image2, *imageOut;
    unsigned w, h, wDs, hDs;
    unsigned subpixels = 4;
    unsigned scaleFactor = 4;
    clock_t start, end;
    double timeElapsed;
    int err;


    // Read images
    start = clock();
    #pragma omp parallel sections
    {
        #pragma omp section 
        readImage(file1, &image1, &w, &h);
        #pragma omp section
        readImage(file2, &image2, &w, &h);
    }
    end = clock();
    timeElapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Read files: %.3f s\n", timeElapsed);

    // OpenCL variables
    cl_uint num_platforms;
    cl_uint num_devices;

    cl_platform_id *platform_id;
    cl_device_id device_id;             // compute device id 
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue

    cl_mem image1GPU, image2GPU, imageDs1GPU, imageDs2GPU, imageGray1GPU, imageGray2GPU, 
        imageZNCC1GPU, imageZNCC2GPU, imageCrossGPU, imageOccGPU, imageOutGPU;

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
 
    // Move images to GPU
    start = clock();
    err  = moveToGPU(image1, &image1GPU, w, h, context, commands);
    err |= moveToGPU(image2, &image2GPU, w, h, context, commands);
    if (err) {
        printf("Error: Failed to move image to the device!\n");
        return 1;
    }
    end = clock();
    timeElapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Move to GPU: %.3f s\n", timeElapsed);

    // Downscale by four
    start = clock();
    err  = downscaleImage(image1GPU, &imageDs1GPU, w, h, subpixels, scaleFactor, context, device_id, commands);
    err |= downscaleImage(image2GPU, &imageDs2GPU, w, h, subpixels, scaleFactor, context, device_id, commands);
    if (err) {
        printf("Error: Failed to downscale image!\n");
        return 1;
    }
    wDs = w / scaleFactor;
    hDs = h / scaleFactor;
    end = clock();
    timeElapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Downscale: %.3f s\n", timeElapsed);

    // Convert to grayscale
    start = clock();
    err  = grayscaleImage(imageDs1GPU, &imageGray1GPU, wDs, hDs, subpixels, context, device_id, commands);
    err |= grayscaleImage(imageDs2GPU, &imageGray2GPU, wDs, hDs, subpixels,  context, device_id, commands);
    if (err) {
        printf("Error: Failed to convert to grayscale!\n");
        return 1;
    }
    end = clock();
    timeElapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Grayscale: %.3f s\n", timeElapsed);

    // Do ZNCC
    start = clock();
    err = calcZNCC(imageGray1GPU, imageGray2GPU, &imageZNCC1GPU, wDs, hDs, MAX_DISP, WIN_SIZE, 1, context, device_id, commands);
    err = calcZNCC(imageGray2GPU, imageGray1GPU, &imageZNCC2GPU, wDs, hDs, MAX_DISP, WIN_SIZE, -1, context, device_id, commands);
    if (err) {
        printf("Error: Failed to calculate ZNCC!\n");
        return 1;
    }
    end = clock();
    timeElapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("ZNCC: %.3f s\n", timeElapsed);

    // Cross checking
    start = clock();
    err = crossCheck(imageZNCC1GPU, imageZNCC2GPU, &imageCrossGPU, wDs, hDs, THRESHOLD, context, device_id, commands);
    if (err) {
        printf("Error: Failed to perform cross checking!\n");
        return 1;
    }
    end = clock();
    timeElapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Cross check: %.3f s\n", timeElapsed);

    // Occlusion fill
    start = clock();
    err = occlusionFill(imageCrossGPU, &imageOccGPU, wDs, hDs, context, device_id, commands);
    if (err) {
        printf("Error: Failed to perform occlusion fill!\n");
        return 1;
    }
    end = clock();
    timeElapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Occlusion fill: %.3f s\n", timeElapsed);

    // Normalize image
    start = clock();
    err = normalizeImage(imageOccGPU, &imageOutGPU, wDs, hDs, context, device_id, commands);
    if (err) {
        printf("Error: Failed to normalize image!\n");
        return 1;
    }
    end = clock();
    timeElapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Normalization: %.3f s\n", timeElapsed);


    size_t region[3] = {wDs, hDs, 1};
    size_t origin[3] = {0, 0, 0};

    imageOut = (unsigned char *) malloc(sizeof(unsigned char) * hDs * wDs);
    clEnqueueReadImage(commands, imageOutGPU, CL_TRUE, origin, region, 0, 0, imageOut, 0, NULL, NULL);

    // Save image
    start = clock();
    writeImage(file3, imageOut, wDs, hDs);
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
    clReleaseMemObject(imageDs2GPU);
    clReleaseMemObject(imageGray1GPU);
    clReleaseMemObject(imageGray2GPU);
    clReleaseMemObject(imageZNCC1GPU);
    clReleaseMemObject(imageZNCC2GPU);

    clReleaseMemObject(imageOutGPU);

    return 0;
}