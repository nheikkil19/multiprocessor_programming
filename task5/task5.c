#include <stdio.h>
#include <stdlib.h>
#include "../lodepng.h"
#include <omp.h>
#include <CL/cl.h>
#include "utils.h"
#include "transformations.h"
#include "disparity.h"

int main(void) {

    // Variables;
    unsigned const MAX_DISP = 260/4;
    unsigned const WIN_SIZE = 9;
    unsigned const THRESHOLD = 8;
    char file1[] = "..\\dataset\\im0.png";
    char file2[] = "..\\dataset\\im1.png";
    char file3[] = "depthmap.png";
    unsigned char *image1, *image2, *imageOut;
    unsigned w, h, wDs, hDs;
    unsigned scaleFactor = 4;
    double start, end, startTotal, endTotal, kernel1, kernel2;
    double timeElapsed, load;
    int err;

    // OpenCL variables
    cl_uint num_platforms;
    cl_uint num_devices;

    cl_platform_id *platform_id;
    cl_device_id device_id;             // compute device id 
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue

    cl_mem image1GPU, image2GPU, imageDs1GPU, imageDs2GPU, imageGray1GPU, imageGray2GPU, 
        imageZNCC1GPU, imageZNCC2GPU, imageCrossGPU, imageOccGPU, imageOutGPU;

    // Print headers
    printf("%-16s %-13s %-13s %-13s %-13s\n", "STEP", "CPU LOAD", "TOTAL TIME", "KERNEL 1", "KERNEL 2");


    // Start timer
    startTotal = getTime();
    getCPULoad();
    // Read images
    start = getTime();
    #pragma omp parallel sections
    {
        #pragma omp section 
        readImage(file1, &image1, &w, &h);
        #pragma omp section
        readImage(file2, &image2, &w, &h);
    }
    load = getCPULoad();
    end = getTime();
    timeElapsed = end - start;
    printf("%-16s %-13.2f %-13f\n", "Read images:", load, timeElapsed);


    // Prepare OpenCL
    start = getTime();
    getCPULoad();
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
    commands = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    if (!commands) {
        printf("Error: Failed to create a command commands!\n");
        return 1;
    }
    load = getCPULoad();
    end = getTime();
    timeElapsed = end - start;
    printf("%-16s %-13.2f %-13f\n", "Prepare OpenCL:", load, timeElapsed);


    // Move images to GPU
    start = getTime();
    getCPULoad();
    err  = moveToGPU(image1, &image1GPU, w, h, context, commands);
    err |= moveToGPU(image2, &image2GPU, w, h, context, commands);
    if (err) {
        printf("Error: Failed to move image to the device!\n");
        return 1;
    }
    load = getCPULoad();
    end = getTime();
    timeElapsed = end - start;
    printf("%-16s %-13.2f %-13f\n", "Move to GPU:", load, timeElapsed);


    // Downscale by four
    start = getTime();
    getCPULoad();
    err  = downscaleImage(image1GPU, &imageDs1GPU, w, h, scaleFactor, context, device_id, commands, &kernel1);
    err |= downscaleImage(image2GPU, &imageDs2GPU, w, h, scaleFactor, context, device_id, commands, &kernel2);
    if (err) {
        printf("Error: Failed to downscale image!\n");
        return 1;
    }
    wDs = w / scaleFactor;
    hDs = h / scaleFactor;
    load = getCPULoad();
    end = getTime();
    timeElapsed = end - start;
    printf("%-16s %-13.2f %-13f %-13f %-13f\n", "Downscale:", load, timeElapsed, kernel1, kernel2);


    // Convert to grayscale
    start = getTime();
    getCPULoad();
    err  = grayscaleImage(imageDs1GPU, &imageGray1GPU, wDs, hDs, context, device_id, commands, &kernel1);
    err |= grayscaleImage(imageDs2GPU, &imageGray2GPU, wDs, hDs, context, device_id, commands, &kernel2);
    if (err) {
        printf("Error: Failed to convert to grayscale!\n");
        return 1;
    }
    load = getCPULoad();
    end = getTime();
    timeElapsed = end - start;
    printf("%-16s %-13.2f %-13f %-13f %-13f\n", "Grayscale:", load, timeElapsed, kernel1, kernel2);


    // Do ZNCC
    start = getTime();
    getCPULoad();
    err = calcZNCC(imageGray1GPU, imageGray2GPU, &imageZNCC1GPU, wDs, hDs, MAX_DISP, WIN_SIZE, 1, context, device_id, commands, &kernel1);
    err = calcZNCC(imageGray2GPU, imageGray1GPU, &imageZNCC2GPU, wDs, hDs, MAX_DISP, WIN_SIZE, -1, context, device_id, commands, &kernel2);
    if (err) {
        printf("Error: Failed to calculate ZNCC!\n");
        return 1;
    }
    load = getCPULoad();
    end = getTime();
    timeElapsed = end - start;
    printf("%-16s %-13.2f %-13f %-13f %-13f\n", "ZNCC:", load, timeElapsed, kernel1, kernel2);


    // Cross checking
    start = getTime();
    getCPULoad();
    err = crossCheck(imageZNCC1GPU, imageZNCC2GPU, &imageCrossGPU, wDs, hDs, THRESHOLD, context, device_id, commands, &kernel1);
    if (err) {
        printf("Error: Failed to perform cross checking!\n");
        return 1;
    }
    load = getCPULoad();
    end = getTime();
    timeElapsed = end - start;
    printf("%-16s %-13.2f %-13f %-13f\n", "Cross check:", load, timeElapsed, kernel1);


    // Occlusion fill
    start = getTime();
    getCPULoad();
    err = occlusionFill(imageCrossGPU, &imageOccGPU, wDs, hDs, context, device_id, commands, &kernel1);
    if (err) {
        printf("Error: Failed to perform occlusion fill!\n");
        return 1;
    }
    load = getCPULoad();
    end = getTime();
    timeElapsed = end - start;
    printf("%-16s %-13.2f %-13f %-13f\n", "Occlusion fill:", load, timeElapsed, kernel1);


    // Normalize image
    start = getTime();
    getCPULoad();
    err = normalizeImage(imageOccGPU, &imageOutGPU, wDs, hDs, context, device_id, commands, &kernel1);
    if (err) {
        printf("Error: Failed to normalize image!\n");
        return 1;
    }
    load = getCPULoad();
    end = getTime();
    timeElapsed = end - start;
    printf("%-16s %-13.2f %-13f %-13f\n", "Normalization:", load, timeElapsed, kernel1);


    // Move image back from GPU
    start = getTime();
    getCPULoad();
    err = moveFromGPU(imageOutGPU, &imageOut, wDs, hDs, commands);
    if (err) {
        printf("Error: Failed to move image from the device!\n");
        return 1;
    }
    load = getCPULoad();
    end = getTime();
    timeElapsed = end - start;
    printf("%-16s %-13.2f %-13f\n", "Move from GPU:", load, timeElapsed);


    // Save image
    start = getTime();
    getCPULoad();
    writeImage(file3, imageOut, wDs, hDs);
    load = getCPULoad();
    end = getTime();
    timeElapsed = end - start;
    printf("%-16s %-13.2f %-13f\n", "Save file:", load, timeElapsed);


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
    clReleaseMemObject(imageCrossGPU);
    clReleaseMemObject(imageOccGPU);
    clReleaseMemObject(imageOutGPU);


    // Report total time
    endTotal = getTime();
    timeElapsed = endTotal - startTotal;
    printf("\nTotal time: %f s\n", timeElapsed);

    // Print device info
    // printf("\n");
    // printDeviceInfo(device_id);

    return 0;
}
