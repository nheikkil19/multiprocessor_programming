

int calcZNCC(cl_mem imageL, cl_mem imageR, cl_mem *imageOut, 
    unsigned w, unsigned h, unsigned max_disp, unsigned win_size, int inv,
    cl_context context, cl_device_id device_id, cl_command_queue commands
) {
    int err = 0;
    size_t global = h;             // global domain size for our calculation
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
    cl_image_format imageFormat;        // image format

    // Set image format
    imageFormat.image_channel_order = CL_A;
    imageFormat.image_channel_data_type = CL_UNSIGNED_INT8;

    // Allocate memory on GPU
    *imageOut = clCreateImage2D(context, CL_MEM_READ_WRITE, &imageFormat, w, h, 0, NULL, &err);

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

    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    if (err) {
        printf("Error: Failed to execute kernel! Error number = %d\n", err);
        return 1;
    }
    clFinish(commands);

    clReleaseProgram(program);
    clReleaseKernel(kernel);
    return 0;
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
