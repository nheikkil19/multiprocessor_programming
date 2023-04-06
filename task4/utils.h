#include <stdio.h>
#include "../lodepng.h"
#include <CL/cl.h>
#include <profileapi.h>

void readImage(char *filename, unsigned char **imageOut, unsigned *w, unsigned *h);
int readTextFile(char *filename, char *content, unsigned maxSize);
void writeImage(char *filename, unsigned char *image, unsigned w, unsigned h);
int moveToGPU(unsigned char *imageIn, cl_mem *imageOut, unsigned w, unsigned h, cl_context context, cl_command_queue commands);
int moveFromGPU(cl_mem imageIn, unsigned char **imageOut, unsigned w, unsigned h, cl_command_queue commands);
void printDeviceInfo(cl_device_id deviceId);
double getTime();
