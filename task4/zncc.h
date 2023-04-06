#include <CL/cl.h>
#include <stdio.h>
#include "utils.h"

int calcZNCC(cl_mem imageL, cl_mem imageR, cl_mem *imageOut, unsigned w, unsigned h, unsigned max_disp, unsigned win_size, int inv, cl_context context, cl_device_id device_id, cl_command_queue commands);
int normalizeImage(cl_mem imageIn, cl_mem *imageOut, unsigned w, unsigned h, cl_context context, cl_device_id device_id, cl_command_queue commands);
int crossCheck(cl_mem image1, cl_mem image2, cl_mem *imageOut, unsigned w, unsigned h, unsigned threshold, cl_context context, cl_device_id device_id, cl_command_queue commands);
int occlusionFill(cl_mem imageIn, cl_mem *imageOut, unsigned w, unsigned h, cl_context context, cl_device_id device_id, cl_command_queue commands);
