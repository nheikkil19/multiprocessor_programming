#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "lodepng.h"
#include <CL/cl.h>

#define W (10000)
#define H (10000)

const char *programSource = "                                   \n" \
"__kernel void addMatrix(                                       \n" \
"   __global unsigned * A,                                      \n" \
"   __global unsigned * B,                                      \n" \
"   __global unsigned * C)                                      \n" \
"   {                                                           \n" \
"    int w, h;                                             \n" \
"    w = 10000;                                                   \n" \
"    h = 10000;                                                   \n" \
"    for (int i=0; i<h; i++) {                                  \n" \
"        for (int j=0; j<w; j++) {                              \n" \
"            C[ w * i + j ] = A[ w * i + j ] + B[ w * i + j ];\n" \
"        }                                                      \n" \
"    }                                                          \n" \
"}                                                              \n" \
"\n";

void addMatrix(unsigned * a, unsigned * b, unsigned * out);
void add_matrix(unsigned * a, unsigned * b, unsigned * out);

int main(void) {

    unsigned * A, * B, * C;
    unsigned w, h;
    w = W;
    h = H;
    A = malloc(w*h*sizeof(unsigned));
    B = malloc(w*h*sizeof(unsigned));
    C = malloc(w*h*sizeof(unsigned));

    for (int i=0; i<h; i++) {
        for (int j=0; j<w; j++) {
            A[ w * i + j ] = 1;
            B[ w * i + j ] = 2;
        }
    }

    // time measurements from geeksforgeeks
    clock_t start, end;
    double timeElapsed;

    start = clock();
    // addMatrix(A, B, C);
    add_matrix(A, B, C);
    end = clock();
    timeElapsed = ((double) (end - start)) / CLOCKS_PER_SEC * 1000;

    printf("Execution time: %f ms\n", timeElapsed);

    free(A);
    free(B);
    free(C);

    return 0;
}

void addMatrix(unsigned * a, unsigned * b, unsigned * out) {

    unsigned w, h;
    w = W;
    h = H;

    for (int i=0; i<h; i++) {
        for (int j=0; j<w; j++) {
            out[ w * i + j ] = a[ w * i + j ] + b[ w * i + j ];
        }
    }

}

void add_matrix(unsigned * a, unsigned * b, unsigned * c) {
    int w = W;
    int h = H;

    int err;
    int gpu = 1;

    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation

    cl_platform_id platform_id;
    cl_device_id device_id;             // compute device id 
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel

    cl_mem A, B;                        // device memory used for the input array
    cl_mem C;                           // device memory used for the output array

    err = clGetPlatformIDs(1, &platform_id, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to get platform id!\n");
    }

    err = clGetDeviceIDs(platform_id, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
        if (err != CL_SUCCESS) {
        printf("Error: Failed to create a device group!\n");
    }

    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context) {
        printf("Error: Failed to create a compute context!\n");
    }

    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands) {
        printf("Error: Failed to create a command commands!\n");
    }

    A = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(unsigned) * h * w, NULL, NULL);
    B = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(unsigned) * h * w, NULL, NULL);
    C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned) * h * w, NULL, NULL);
    if (!A || !B || !C) {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }

    err = clEnqueueWriteBuffer(commands, A, CL_TRUE, 0, sizeof(unsigned) * h * w, a, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commands, B, CL_TRUE, 0, sizeof(unsigned) * h * w, b, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }

    program = clCreateProgramWithSource(context, 1, (const char **) &programSource, NULL, &err);
    if (!program) {
        printf("Error: Failed to create compute program!\n");
    }

    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }

    kernel = clCreateKernel(program, "addMatrix", &err);
    if (!kernel || err != CL_SUCCESS) {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &A);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &B);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &C);
    // err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &count);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }

    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }

    global = 1024;
    local = 64;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err) {
        printf("Error: Failed to execute kernel!\n");
    }

    clFinish(commands);

    err = clEnqueueReadBuffer( commands, C, CL_TRUE, 0, sizeof(unsigned) * h * w, c, 0, NULL, NULL );  
    if (err != CL_SUCCESS) {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }

    clFlush(commands);


    clReleaseMemObject(A);
    clReleaseMemObject(B);
    clReleaseMemObject(C);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

}