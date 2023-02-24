#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "lodepng.h"
#include <CL/cl.h>

int convertGrayscale(unsigned char *image, unsigned w, unsigned h, unsigned p, unsigned char *imageOut, cl_device_id device_id, cl_context context, cl_command_queue commands);

int applyFilter(unsigned char *image, unsigned w, unsigned h, unsigned char *imageOut, cl_device_id device_id, cl_context context, cl_command_queue commands);


int main(void) {

    // Read image. from lodepng examples
    unsigned char *image;
    unsigned w, h, p = 4;
    unsigned err;
    char filenameIn[] = "dataset\\im2.png";
    char filenameOut[] = "dataset\\im2_gray.png";
    


    // OpenCL variables
    // size_t global;                      // global domain size for our calculation
    // size_t local;                       // local domain size for our calculation

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
        printf("Error: Failed to get platform id!\n");
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
    unsigned char *imageGray = (unsigned char *) malloc(sizeof(unsigned char) * w * h);
    err = convertGrayscale(image, w, h, p, imageGray, device_id, context, commands);
    if (err) {
        printf("Error: Failed to convert image to grayscale!\n");
        return 1;
    }

    // LodePNGColorType colorType;
    // colorType = LCT_GREY;
    // err = lodepng_encode_file(filenameOut, (unsigned char *) imageGray, w, h, colorType, 8);
    // if (err) {
    //     printf("Error %u\n", err);
    // }


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


    return 0;
}


int convertGrayscale(unsigned char *image, unsigned w, unsigned h, unsigned p, unsigned char *imageOut, cl_device_id device_id, cl_context context, cl_command_queue commands) {

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

    // err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    // if (err != CL_SUCCESS) {
    //     printf("Error: Failed to retrieve kernel work group info! %d\n", err);
    //     exit(1);
    // }

    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err) {
        printf("Error: Failed to execute kernel!\n");
        return 1;
    }
    clFinish(commands);

    // unsigned char *temp = (unsigned char * ) malloc(sizeof(unsigned char) * w *h);
    err = clEnqueueReadBuffer( commands, clImageOut, CL_TRUE, 0, sizeof(unsigned char) * w* h, imageOut, 0, NULL, NULL );
    if (err != CL_SUCCESS) {
        printf("Error: Failed to read output array! %d\n", err);
        return 1;
    }
    // free(temp);

    // *imageOut = clImageOut;

    clFlush(commands);

    clReleaseMemObject(clImageIn);
    clReleaseProgram(program);
    clReleaseKernel(kernel);

    return 0;
}

int applyFilter(unsigned char *image, unsigned w, unsigned h, unsigned char *imageOut, cl_device_id device_id, cl_context context, cl_command_queue commands) {

    int err = 0;
    size_t global = 1024;               // global domain size for our calculation
    size_t local = 64;                  // local domain size for our calculation
    cl_mem clImageIn, clImageOut, clFilter;
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel

    // create filter
    unsigned filterdim = 5;
    unsigned filtersize = filterdim * filterdim;
    float *filter = (float *) malloc(sizeof(float) * filtersize);
    for (int i=0; i<filtersize; i++) {
        filter[i] = (float) 1/filtersize;
    }

    char *programSource = "     \n" \
        "__kernel void applyFilter(\n" \
        "    __global unsigned char *inImage,\n" \
        "    __global unsigned char *outImage,\n" \
        "    __global float *filter,\n" \
        "    unsigned w,\n" \
        "    unsigned h,\n" \
        "    unsigned filterdim\n" \
        ") {\n" \
        "   int mid = filterdim / 2;\n" \
        "   int xx, yy;\n" \
        "   float sum;\n" \
        "   for (int i=0; i<h; i++) {\n" \
        "       for (int j=0; j<w; j++) {\n" \
        "           sum = 0;\n" \
        "           for (int y=0; y<filterdim; y++) {\n" \
        "               yy = i+y-mid;\n" \
        "               if ( yy >= 0 && yy < h ) {\n" \
        "                   for (int x=0; x<filterdim; x++) {\n" \
        "                       xx = j+x-mid;\n" \
        "                       if ( xx >= 0 && xx < w ) {\n" \
        "                           sum += inImage[w * yy + xx] *\n" \
        "                           filter[ filterdim*y + x];\n" \
        "                       }\n" \
        "                    }\n" \
        "                }\n" \
        "            }\n" \
        "            outImage[ w*i + j ] = (unsigned char) sum;\n" \
        "        }\n" \
        "    }\n" \
        "}";


    // clImageIn = *image;
    clImageIn = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * h * w, NULL, NULL);
    clImageOut = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * h * w, NULL, NULL);
    clFilter = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * filtersize, NULL, NULL);
    if (!clImageIn || !clImageOut || !clFilter) {
        printf("Error: Failed to allocate device memory!\n");
        return 1;
    }

    err = clEnqueueWriteBuffer(commands, clFilter, CL_TRUE, 0, sizeof(float) * filtersize, filter, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commands, clImageIn, CL_TRUE, 0, sizeof(unsigned char) * w*h, image, 0, NULL, NULL);
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

    // err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    // if (err != CL_SUCCESS) {
    //     printf("Error: Failed to retrieve kernel work group info! %d\n", err);
    //     exit(1);
    // }

    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err) {
        printf("Error: Failed to execute kernel!\n");
        return 1;
    }

    clFinish(commands);
    // float * temp = (float *) malloc(sizeof(float) * w * h);
    // err = clEnqueueReadBuffer( commands, clImageOut, CL_TRUE, 0, sizeof(float) * w * h, temp, 0, NULL, NULL );
    err = clEnqueueReadBuffer( commands, clImageOut, CL_TRUE, 0, sizeof(unsigned char) * w * h, imageOut, 0, NULL, NULL );
    if (err != CL_SUCCESS) {
        printf("Error: Failed to read output array! %d\n", err);
        return 1;
    }
    // free(temp);
    clFlush(commands);

    clReleaseMemObject(clImageIn);
    clReleaseMemObject(clImageOut);
    clReleaseMemObject(clFilter);
    clReleaseProgram(program);
    clReleaseKernel(kernel);

    free(filter);
    return 0;
}


// void applyFilter(
//     int *inImage,
//     float *outImage,
//     float *filter,
//     unsigned w,
//     unsigned h,
//     unsigned filterdim
// ) {
//     int mid = filterdim / 2;
//     int xx, yy;
//     for (int i=0; i<h; i++) {
//         for (int j=0; j<w; j++) {
//             outImage[ w*i + j ] = 0;
//             for (int y=0; y<filterdim; y++) {
//                 yy = i+y-mid;
//                 if ( yy >= 0 && yy < h ) {
//                     for (int x=0; x<filterdim; x++) {
//                         xx = j+x-mid;
//                         if ( xx >= 0 && xx < w ) {
//                             outImage[ w*i + j ] += inImage[w * yy + xx] *
//                             filter[ filterdim*y + x];
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }