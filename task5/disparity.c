#include "disparity.h"

int calcZNCC(cl_mem imageL, cl_mem imageR, cl_mem *imageOut, 
    unsigned w, unsigned h, unsigned max_disp, unsigned win_size, int inv,
    cl_context context, cl_device_id device_id, cl_command_queue commands,
    double *time
) {
    int err = 0;
    size_t global[2] = {ceil(h/4.f)*4, ceil(w/8.f)*8};          // total number of work-items in each dimension
    size_t local[2] = {4, 8};
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
    cl_event event;                     // command queue event
    cl_ulong start, end;                // kernel execution time measurements

    // Allocate memory on GPU
    *imageOut = clCreateBuffer(context, CL_MEM_READ_WRITE, w * h * sizeof(unsigned char), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to create buffer on device!\n");
        return 1;
    }

    // Read the kernel code from file
    char *programSource;
    programSource = (char *) malloc(sizeof(char) * 4096);
    err = readTextFile("zncc.cl", programSource, 4096);
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

    kernel = clCreateKernel(program, "calcZNCC", &err);
    if (!kernel || err != CL_SUCCESS) {
        printf("Error: Failed to create compute kernel!\n");
        return 1;
    }

    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &imageL);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &imageR);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), imageOut);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned), &w);
    err |= clSetKernelArg(kernel, 4, sizeof(unsigned), &h);
    err |= clSetKernelArg(kernel, 5, sizeof(unsigned), &max_disp);
    err |= clSetKernelArg(kernel, 6, sizeof(unsigned), &win_size);
    err |= clSetKernelArg(kernel, 7, sizeof(int), &inv);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        return 1;
    }

    err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, global, local, 0, NULL, &event);
    if (err) {
        printf("Error: Failed to execute kernel! Error number = %d\n", err);
        return 1;
    }
    clFinish(commands);

    err = clGetEventProfilingInfo(event,  CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    if (err != CL_SUCCESS) {
        printf("Error %d: Failed to get the start timer!\n", err);
        return 1;
    }
    err = clGetEventProfilingInfo(event,  CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    if (err != CL_SUCCESS) {
        printf("Error %d: Failed to end timer!\n", err);
        return 1;
    }
    *time = (double) (end-start) / 1000000000;

    clReleaseProgram(program);
    clReleaseKernel(kernel);
    return 0;
}


int normalizeImage(cl_mem imageIn, cl_mem *imageOut, unsigned w, unsigned h,
    cl_context context, cl_device_id device_id, cl_command_queue commands,
    double *time
) {
    int err = 0;
    size_t global = w;                  // global domain size for our calculation
    size_t local = w;
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
    cl_event event;                     // command queue event
    cl_ulong start, end;                // kernel execution time measurements


    // Allocate memory on GPU
    *imageOut = clCreateBuffer(context, CL_MEM_WRITE_ONLY, w * h * sizeof(unsigned char), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to create buffer on device!\n");
        return 1;
    }

    // Read the kernel code from file
    char *programSource;
    programSource = (char *) malloc(sizeof(char) * 2048);
    err = readTextFile("normalize.cl", programSource, 2048);
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

    kernel = clCreateKernel(program, "normalizeImage", &err);
    if (!kernel || err != CL_SUCCESS) {
        printf("Error: Failed to create compute kernel!\n");
        return 1;
    }

    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &imageIn);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), imageOut);
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned), &w);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned), &h);
    err |= clSetKernelArg(kernel, 4, sizeof(int) * w * 2, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        return 1;
    }

    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, &event);
    if (err) {
        printf("Error: Failed to execute kernel! Error number = %d\n", err);
        return 1;
    }
    clFinish(commands);

    err = clGetEventProfilingInfo(event,  CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    if (err != CL_SUCCESS) {
        printf("Error %d: Failed to get the start timer!\n", err);
        return 1;
    }
    err = clGetEventProfilingInfo(event,  CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    if (err != CL_SUCCESS) {
        printf("Error %d: Failed to end timer!\n", err);
        return 1;
    }
    *time = (double) (end-start) / 1000000000;

    clReleaseProgram(program);
    clReleaseKernel(kernel);
    return 0;
}


int crossCheck(cl_mem image1, cl_mem image2, cl_mem *imageOut, 
    unsigned w, unsigned h, unsigned threshold,
    cl_context context, cl_device_id device_id, cl_command_queue commands,
    double *time
) {
    int err = 0;
    size_t global[2] = {ceil(h/8.f)*8, ceil(w/8.f)*8};          // total number of work-items in each dimension
    size_t local[2] = {8, 8};
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
    cl_event event;                     // command queue event
    cl_ulong start, end;                // kernel execution time measurements


    // Allocate memory on GPU
    *imageOut = clCreateBuffer(context, CL_MEM_READ_WRITE, w * h * sizeof(unsigned char), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to create buffer on device!\n");
        return 1;
    }

    // Read the kernel code from file
    char *programSource;
    programSource = (char *) malloc(sizeof(char) * 2048);
    err = readTextFile("crosscheck.cl", programSource, 2048);
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

    kernel = clCreateKernel(program, "crossCheck", &err);
    if (!kernel || err != CL_SUCCESS) {
        printf("Error: Failed to create compute kernel!\n");
        return 1;
    }

    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &image1);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &image2);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), imageOut);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned), &w);
    err |= clSetKernelArg(kernel, 4, sizeof(unsigned), &h);
    err |= clSetKernelArg(kernel, 5, sizeof(unsigned), &threshold);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        return 1;
    }

    err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, global, local, 0, NULL, &event);
    if (err) {
        printf("Error: Failed to execute kernel! Error number = %d\n", err);
        return 1;
    }
    clFinish(commands);

    err = clGetEventProfilingInfo(event,  CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    if (err != CL_SUCCESS) {
        printf("Error %d: Failed to get the start timer!\n", err);
        return 1;
    }
    err = clGetEventProfilingInfo(event,  CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    if (err != CL_SUCCESS) {
        printf("Error %d: Failed to end timer!\n", err);
        return 1;
    }
    *time = (double) (end-start) / 1000000000;

    clReleaseProgram(program);
    clReleaseKernel(kernel);
    return 0;
}

int occlusionFill(cl_mem imageIn, cl_mem *imageOut, unsigned w, unsigned h,
    cl_context context, cl_device_id device_id, cl_command_queue commands,
    double *time
) {
    int err = 0;
    size_t global[2] = {ceil(h/8.f)*8, ceil(w/8.f)*8};          // total number of work-items in each dimension
    size_t local[2] = {8, 8};
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
    cl_event event;                     // command queue event
    cl_ulong start, end;                // kernel execution time measurements

    // Allocate memory on GPU
    *imageOut = clCreateBuffer(context, CL_MEM_READ_WRITE, w * h * sizeof(unsigned char), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to create buffer on device!\n");
        return 1;
    }

    // Read the kernel code from file
    char *programSource;
    programSource = (char *) malloc(sizeof(char) * 2048);
    err = readTextFile("occlusionfill.cl", programSource, 2048);
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

    kernel = clCreateKernel(program, "occlusionFill", &err);
    if (!kernel || err != CL_SUCCESS) {
        printf("Error: Failed to create compute kernel!\n");
        return 1;
    }

    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &imageIn);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), imageOut);
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned), &w);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned), &h);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        return 1;
    }

    err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, global, local, 0, NULL, &event);
    if (err) {
        printf("Error: Failed to execute kernel! Error number = %d\n", err);
        return 1;
    }
    clFinish(commands);

    err = clGetEventProfilingInfo(event,  CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    if (err != CL_SUCCESS) {
        printf("Error %d: Failed to get the start timer!\n", err);
        return 1;
    }
    err = clGetEventProfilingInfo(event,  CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    if (err != CL_SUCCESS) {
        printf("Error %d: Failed to end timer!\n", err);
        return 1;
    }
    *time = (double) (end-start) / 1000000000;

    clReleaseProgram(program);
    clReleaseKernel(kernel);
    return 0;
}
