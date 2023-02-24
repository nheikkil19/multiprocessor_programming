#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "lodepng.h"
#include <CL/cl.h>

int convertGrayscale(unsigned char *image, unsigned w, unsigned h, unsigned p, cl_mem *imageOut, cl_device_id device_id, cl_context context, cl_command_queue commands);

int applyFilter(cl_mem *image, unsigned w, unsigned h, unsigned char *imageOut, cl_device_id device_id, cl_context context, cl_command_queue commands);

void printSystemInfo(cl_device_id device_id, cl_platform_id *platform_id, cl_uint num_platforms, cl_uint num_devices);

int main(void) {

    // Variables
    unsigned char *image;
    unsigned w, h, p = 4;
    unsigned err;
    char filenameIn[] = "dataset\\im0.png";
    char filenameOut[] = "dataset\\im0_gray.png";

    cl_uint num_platforms;
    cl_uint num_devices;
    cl_uint gpu = 1;                    // use gpu

    cl_platform_id *platform_id;
    cl_device_id device_id;             // compute device id 
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue



    // =========================================
    // Read image
    err = lodepng_decode32_file(&image, &w, &h, filenameIn);
    if (err) {
        printf("Error %u\n", err);
        return 1;
    }
    else {
        printf("w = %u, h = %u\n", w, h);
    }

    // =======================================
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

    err = clGetDeviceIDs(platform_id[0], gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, &num_devices);
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

    // ========================================
    // Convert to grayscale
    cl_mem *imageGray = (cl_mem *) malloc(sizeof(cl_mem));
    err = convertGrayscale(image, w, h, p, imageGray, device_id, context, commands);
    if (err) {
        printf("Error: Failed to convert image to grayscale!\n");
        return 1;
    }

    // ============================================
    // Apply filter
    unsigned char *imageOut = (unsigned char *) malloc(sizeof(unsigned char) * w * h);
    err = applyFilter(imageGray, w, h, imageOut, device_id, context, commands);
    if (err) {
        printf("Error: Failed to apply filter!\n");
        return 1;
    }

    // ==============================================
    // Encode to file
    LodePNGColorType colorType;
    colorType = LCT_GREY;
    err = lodepng_encode_file(filenameOut, imageOut, w, h, colorType, 8);
    if (err) {
        printf("Error %u\n", err);
    }

    // ================================
    // Free variables
    free(image);
    free(imageOut);
    free(imageGray);

    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    printSystemInfo(device_id, platform_id, num_platforms, num_devices);
    return 0;
}


int convertGrayscale(unsigned char *image, unsigned w, unsigned h, unsigned p, cl_mem *imageOut, cl_device_id device_id, cl_context context, cl_command_queue commands) {

    int err = 0;
    size_t global = 1024;               // global domain size for our calculation
    size_t local = 64;                  // local domain size for our calculation
    cl_mem clImageIn, clImageOut;
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel

    char *programSource = "     \n" \
        "__kernel void convertGrayscale(            \n" \
        "    __global unsigned char *inImage,       \n" \
        "    __global unsigned char *outImage,      \n" \
        "    unsigned w,                   \n" \
        "    unsigned h,                   \n" \
        "    unsigned p                \n" \
        ") {                            \n" \
        "    for (int i=0; i<h; i++) {                  \n" \
        "        for (int j=0; j<w; j++) {              \n" \
        "            unsigned char r, g, b;             \n" \
        "            r = inImage[ w*p*i + j*p + 0 ];    \n" \
        "            g = inImage[ w*p*i + j*p + 1 ];    \n" \
        "            b = inImage[ w*p*i + j*p + 2 ];    \n" \
        "            outImage[ w*i + j ] = 0.2126 * r + 0.7152 * g + 0.0722 * b;    \n" \
        "        }  \n" \
        "    }      \n" \
        "}";


    clImageIn = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(unsigned char) * h * w * p, NULL, NULL);
    clImageOut = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * h * w, NULL, NULL);
    if (!clImageIn || !clImageOut) {
        printf("Error: Failed to allocate device memory!\n");
        return 1;
    }

    err = clEnqueueWriteBuffer(commands, clImageIn, CL_TRUE, 0, sizeof(unsigned char) * h * w * p, image, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to write to source array!\n");
        return 1;
    }

    program = clCreateProgramWithSource(context, 1, (const char **) &programSource, NULL, &err);
    if (!program) {
        printf("Error: Failed to create compute program!\n");
        return 1;
    }

    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return 1;
    }

    kernel = clCreateKernel(program, "convertGrayscale", &err);
    if (!kernel || err != CL_SUCCESS) {
        printf("Error: Failed to create compute kernel!\n");
        return 1;
    }

    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &clImageIn);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &clImageOut);
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned), &w);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned), &h);
    err |= clSetKernelArg(kernel, 4, sizeof(unsigned), &p);
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

    *imageOut = clImageOut;

    clFlush(commands);

    clReleaseMemObject(clImageIn);
    clReleaseProgram(program);
    clReleaseKernel(kernel);

    return 0;
}

int applyFilter(cl_mem *image, unsigned w, unsigned h, unsigned char *imageOut, cl_device_id device_id, cl_context context, cl_command_queue commands) {

    int err = 0;
    size_t global = 1024;               // global domain size for our calculation
    size_t local = 64;                  // local domain size for our calculation
    cl_mem clImageIn, clImageOut, clFilter;
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel

    // Create filter
    unsigned filterdim = 5;
    unsigned filtersize = filterdim * filterdim;
    float *filter = (float *) malloc(sizeof(float) * filtersize);
    for (int i=0; i<filtersize; i++) {
        filter[i] = (float) 1/filtersize;
    }

    char *programSource = "     \n" \
        "__kernel void applyFilter(             \n" \
        "    __global unsigned char *inImage,   \n" \
        "    __global unsigned char *outImage,  \n" \
        "    __global float *filter,            \n" \
        "    unsigned w,                        \n" \
        "    unsigned h,                        \n" \
        "    unsigned filterdim                 \n" \
        ") {                                    \n" \
        "   int mid = filterdim / 2;            \n" \
        "   int xx, yy;                         \n" \
        "   float sum;                          \n" \
        "   for (int i=0; i<h; i++) {           \n" \
        "       for (int j=0; j<w; j++) {       \n" \
        "           sum = 0;                            \n" \
        "           for (int y=0; y<filterdim; y++) {   \n" \
        "               yy = i+y-mid;                   \n" \
        "               if ( yy >= 0 && yy < h ) {                  \n" \
        "                   for (int x=0; x<filterdim; x++) {       \n" \
        "                       xx = j+x-mid;                       \n" \
        "                       if ( xx >= 0 && xx < w ) {          \n" \
        "                           sum += inImage[w * yy + xx] *   \n" \
        "                           filter[ filterdim*y + x];       \n" \
        "                       }       \n" \
        "                    }          \n" \
        "                }              \n" \
        "            }                  \n" \
        "            outImage[ w*i + j ] = (unsigned char) sum;     \n" \
        "        }          \n" \
        "    }              \n" \
        "}";


    clImageIn = *image;
    clImageOut = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * h * w, NULL, NULL);
    clFilter = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * filtersize, NULL, NULL);
    if (!clImageIn || !clImageOut || !clFilter) {
        printf("Error: Failed to allocate device memory!\n");
        return 1;
    }

    err = clEnqueueWriteBuffer(commands, clFilter, CL_TRUE, 0, sizeof(float) * filtersize, filter, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to write to source array!\n");
        return 1;
    }

    program = clCreateProgramWithSource(context, 1, (const char **) &programSource, NULL, &err);
    if (!program) {
        printf("Error: Failed to create compute program!\n");
        return 1;
    }

    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return 1;
    }

    kernel = clCreateKernel(program, "applyFilter", &err);
    if (!kernel || err != CL_SUCCESS) {
        printf("Error: Failed to create compute kernel!\n");
        return 1;
    }

    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &clImageIn);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &clImageOut);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &clFilter);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned), &w);
    err |= clSetKernelArg(kernel, 4, sizeof(unsigned), &h);
    err |= clSetKernelArg(kernel, 5, sizeof(unsigned), &filterdim);
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

    err = clEnqueueReadBuffer( commands, clImageOut, CL_TRUE, 0, sizeof(unsigned char) * w * h, imageOut, 0, NULL, NULL );
    if (err != CL_SUCCESS) {
        printf("Error: Failed to read output array! %d\n", err);
        return 1;
    }

    clFlush(commands);

    clReleaseMemObject(clImageIn);
    clReleaseMemObject(clImageOut);
    clReleaseMemObject(clFilter);
    clReleaseProgram(program);
    clReleaseKernel(kernel);

    free(filter);
    return 0;
}


void printSystemInfo(cl_device_id device_id, cl_platform_id *platform_id, cl_uint num_platforms, cl_uint num_devices) {
    size_t max_size = 100;
    char * platform_name = (char *) malloc(sizeof(char)*max_size);
    char * device_name = (char *) malloc(sizeof(char)*max_size);
    char * driver_version = (char *) malloc(sizeof(char)*max_size);
    char * opencl_c_version = (char *) malloc(sizeof(char)*max_size);
    cl_uint compute_units;
    cl_uint max_work_item_dimensions;
    clGetPlatformInfo(platform_id[0], CL_PLATFORM_VERSION, max_size, platform_name, NULL);
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, max_size, device_name, NULL);
    clGetDeviceInfo(device_id, CL_DRIVER_VERSION, max_size, driver_version, NULL);
    clGetDeviceInfo(device_id, CL_DEVICE_OPENCL_C_VERSION, max_size, opencl_c_version, NULL);
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &compute_units, NULL);
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &max_work_item_dimensions, NULL);

    printf("Platform Count: %d \n", num_platforms);
    printf("Device Count on Platform 1: %d \n", num_devices);
    printf("Device: %s \n", device_name);
    printf("Hardware version: %s \n", platform_name);
    printf("Driver version: %s \n", driver_version);
    printf("OpenCL C version: %s \n", opencl_c_version);
    printf("Parallel Compute units: %u \n", compute_units);
    printf("Max Work Item Dimensions: %u \n", max_work_item_dimensions);

    free(platform_name);
    free(device_name);
    free(driver_version);
    free(opencl_c_version);
}