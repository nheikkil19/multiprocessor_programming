#include <CL/cl.h>
#include <stdio.h>
#include "utils.h"

int downscaleImage(cl_mem imageIn, cl_mem *imageOut, unsigned w, unsigned h, unsigned subpixels, unsigned factor, cl_context context, cl_device_id device_id, cl_command_queue commands);
int grayscaleImage(cl_mem imageIn, cl_mem *imageOut, unsigned w, unsigned h, unsigned subpixels, cl_context context, cl_device_id device_id, cl_command_queue commands);
