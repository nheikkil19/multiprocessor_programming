#include "utils.h"
#


void readImage(char *filename, unsigned char **imageOut, unsigned *w, unsigned *h) {
    int err;
    err = lodepng_decode32_file(imageOut, w, h, filename);
    if (err) {
        printf("Error: Error when reading the image.");
    }
}

int readTextFile(char *filename, char *content, unsigned maxSize) {
    FILE *file;
    file = fopen(filename, "r");
    char c;
    if (file == NULL) {
        return 1;
    }
    else {
        int i = 0;
        while ((c = fgetc(file)) != EOF && i < maxSize-1) {
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


int moveToGPU(unsigned char *imageIn, cl_mem *imageOut, unsigned w, unsigned h, 
    cl_context context, cl_command_queue commands
) {
    cl_image_format imageFormat;
    cl_image_desc imageDesc;
    int err;

    // Set image format
    imageFormat.image_channel_order = CL_RGBA;
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

    // Move images to GPU
    size_t origin[] = {0, 0, 0};
    size_t region[] = {w, h, 1};
    err  = clEnqueueWriteImage(commands, *imageOut, CL_TRUE, origin, region, 0, 0, imageIn, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to write image to GPU memory!\n");
        return 1;
    }
    return 0;

}

int moveFromGPU(cl_mem imageIn, unsigned char **imageOut, unsigned w, unsigned h, 
    cl_command_queue commands
) {
    int err;

    // Allocate memory on host
    *imageOut = (unsigned char *) malloc(w * h * sizeof(unsigned char));

    // Move images from GPU
    size_t origin[] = {0, 0, 0};
    size_t region[] = {w, h, 1};
    err  = clEnqueueReadImage(commands, imageIn, CL_TRUE, origin, region, 0, 0, *imageOut, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to read image from GPU memory!\n");
        return 1;
    }
    return 0;
}

void printDeviceInfo(cl_device_id deviceId) {
    cl_device_local_mem_type localMemType;
    cl_ulong localMemSize;
    cl_uint maxComputeUnits;
    cl_uint maxClockFrequency;
    cl_ulong maxConstantBufferSize;
    size_t maxWorkGroupSize;
    unsigned maxWorkItemDimensions;

    clGetDeviceInfo(deviceId, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(localMemType), &localMemType, NULL);
    clGetDeviceInfo(deviceId, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMemSize), &localMemSize, NULL);
    clGetDeviceInfo(deviceId, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, NULL);
    clGetDeviceInfo(deviceId, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(maxClockFrequency), &maxClockFrequency, NULL);
    clGetDeviceInfo(deviceId, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(maxConstantBufferSize), &maxConstantBufferSize, NULL);
    clGetDeviceInfo(deviceId, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);
    clGetDeviceInfo(deviceId, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(maxWorkItemDimensions), &maxWorkItemDimensions, NULL);

    size_t maxWorkItemSizes[maxWorkItemDimensions];
    clGetDeviceInfo(deviceId, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(maxWorkItemSizes), maxWorkItemSizes, NULL);


    printf("Local memory type: ");
    if (localMemType == CL_LOCAL) {
        printf("Local\n");
    }
    else if (localMemType == CL_GLOBAL) {
        printf("Global\n");
    }
    else {
        printf("Not supported\n");
    }

    printf("Local memory size: %llu B\n", localMemSize);
    printf("Max compute units: %d\n", maxComputeUnits);
    printf("Max clock frequency: %d MHz\n", maxClockFrequency);
    printf("Max constant buffer size: %llu B\n", maxConstantBufferSize);
    printf("Max work group size: %llu \n", maxWorkGroupSize);
    printf("Nax work item dimensions: %d \n", maxWorkItemDimensions);
    printf("Max work item sizes: (");
    for (int i = 0; i < maxWorkItemDimensions-1; i++) {
        printf("%llu, ", maxWorkItemSizes[i]);
    }
    printf("%llu) \n", maxWorkItemSizes[maxWorkItemDimensions-1]);

}


double getTime() {
    LARGE_INTEGER freq, ticks;
    static long long frequency = 0; 

    if (frequency == 0) {
        QueryPerformanceFrequency(&freq);
        frequency = freq.QuadPart;
    }

    QueryPerformanceCounter(&ticks);

    return (double) ticks.QuadPart / frequency;
}
