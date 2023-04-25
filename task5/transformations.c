#include "transformations.h"

int downscaleImage(cl_mem imageIn, cl_mem *imageOut, unsigned w, unsigned h, 
    unsigned subpixels, unsigned factor, 
    cl_context context, cl_device_id device_id, cl_command_queue commands
) {
    int err = 0;
    size_t global[3] = {h/factor, w/factor, subpixels};           // total number of work-items in each dimension
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
    cl_image_format imageFormat;        // image format
    cl_image_desc imageDesc;            // image descriptor
    cl_event event;                     // command queue event
    cl_ulong start, end;                // kernel execution time measurements

    // Set output image format
    imageFormat.image_channel_order = CL_RGBA;
    imageFormat.image_channel_data_type = CL_UNSIGNED_INT8;

    // Set image descriptor
    imageDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    imageDesc.image_width = w/factor;
    imageDesc.image_height = h/factor;
    imageDesc.image_depth = 0;
    imageDesc.image_array_size = 1;
    imageDesc.image_row_pitch = 0;
    imageDesc.image_slice_pitch = 0;
    imageDesc.num_mip_levels = 0;
    imageDesc.num_samples = 0;
    imageDesc.buffer = NULL;

    // Allocate memory on GPU
    *imageOut = clCreateImage(context, CL_MEM_READ_WRITE, &imageFormat, &imageDesc, NULL, &err);
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
    err = clEnqueueNDRangeKernel(commands, kernel, 3, NULL, global, NULL, 0, NULL, &event);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to execute kernel!\n");
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
    printf("Downscale kernel: %f s\n", (double) (end-start) / 1000000000);

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
    cl_image_desc imageDesc;
    cl_event event;                     // command queue event
    cl_ulong start, end;                // kernel execution time measurements

    // Set output image format
    imageFormat.image_channel_order = CL_R;
    imageFormat.image_channel_data_type = CL_UNSIGNED_INT8;

    // Set image descriptor
    imageDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    imageDesc.image_width = w;
    imageDesc.image_height = h;
    imageDesc.image_depth = 0;
    imageDesc.image_array_size = 1;
    imageDesc.image_row_pitch = 0;
    imageDesc.image_slice_pitch = 0;
    imageDesc.num_mip_levels = 0;
    imageDesc.num_samples = 0;
    imageDesc.buffer = NULL;

    // Allocate memory on GPU
    *imageOut = clCreateImage(context, CL_MEM_READ_WRITE, &imageFormat, &imageDesc, NULL, &err);
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
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, NULL, 0, NULL, &event);
    if (err) {
        printf("Error: Failed to execute kernel!\n");
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
    printf("Grayscale kernel: %f s\n", (double) (end-start) / 1000000000);

    clReleaseProgram(program);
    clReleaseKernel(kernel);
    return 0;
}
