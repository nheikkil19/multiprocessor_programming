#include <CL/cl.h>

int downscaleImage(cl_mem imageIn, cl_mem *imageOut, unsigned w, unsigned h, 
    unsigned subpixels, unsigned factor, 
    cl_context context, cl_device_id device_id, cl_command_queue commands
) {
    int err = 0;
    size_t global = h/factor;               // global domain size for our calculation
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
    cl_image_format imageFormat;        // image format

    // Set output image format
    imageFormat.image_channel_order = CL_RGBA;
    imageFormat.image_channel_data_type = CL_UNSIGNED_INT8;

    // Allocate memory on GPU
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
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    if (err) {
        printf("Error: Failed to execute kernel!\n");
        return 1;
    }
    clFinish(commands);

    clReleaseProgram(program);
    clReleaseKernel(kernel);
    return 0;
}

int grayscaleImage(cl_mem imageIn, cl_mem *imageOut,
    unsigned w, unsigned h, unsigned subpixels,
    cl_context context, cl_device_id device_id, cl_command_queue commands
) {
    int err = 0;
    size_t global = h;               // global domain size for our calculation
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
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    if (err) {
        printf("Error: Failed to execute kernel!\n");
        return 1;
    }
    clFinish(commands);

    clReleaseProgram(program);
    clReleaseKernel(kernel);
    return 0;
}
