#include <stdio.h>
#include <stdlib.h>
#include "lodepng.h"

void readImage(char *filename, unsigned char **imageOut, unsigned *w, unsigned *h) {
    int err;
    err = lodepng_decode32_file(imageOut, w, h, filename);
    if (err) {
        printf("Error: Error when reading the image.");
    }
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

void downscaleImage(unsigned char *imageIn, unsigned char **imageOut, unsigned w, unsigned h, unsigned factor) {
    unsigned hOut, wOut;
    unsigned subpixels = 4;
    hOut = h / factor;
    wOut = w / factor;
    *imageOut = (unsigned char *) malloc(sizeof(unsigned char) * wOut * hOut * subpixels);

    for (int i=0; i<hOut; i++) {
        for (int j=0; j<wOut; j++) {
            for (int k=0; k<subpixels; k++) {
                (*imageOut)[i*wOut*subpixels + j*subpixels + k] = imageIn[i*factor*w*subpixels + j*factor*subpixels + k];
            }
        }
    }
}

void grayscaleImage(unsigned char *imageIn, unsigned char **imageOut, unsigned w, unsigned h) {
    unsigned subpixels = 4;
    unsigned r, g, b;
    *imageOut = (unsigned char *) malloc(sizeof(unsigned char) * w * h);
    for (int i=0; i<h; i++) {
        for (int j=0; j<w; j++) {
            r = imageIn[i*w*subpixels + j*subpixels + 0];
            g = imageIn[i*w*subpixels + j*subpixels + 1];
            b = imageIn[i*w*subpixels + j*subpixels + 2];
            (*imageOut)[i*w + j] = (int) (0.2126 * r + 0.7152 * g + 0.0722 * b);
        }
    }
}

int main(void) {

    // Init variables
    char file1[] = "dataset\\im0.png";
    char file2[] = "dataset\\im1.png";
    char file3[] = "dataset\\im_gray.png";
    unsigned char *image1, *image2, *imageOut, *imageOut2;
    unsigned w, h, scaleFactor;

    // Read images
    readImage(file1, &image1, &w, &h);

    // Downscale by four
    scaleFactor = 4;
    downscaleImage(image1, &imageOut, w, h, scaleFactor);
    w = w / scaleFactor;
    h = h / scaleFactor;

    // Convert to grayscale
    grayscaleImage(imageOut, &imageOut2, w, h);

    // Do ZNCC

    // Save image
    writeImage(file3, imageOut2, w, h);

    // Free memory
    free(image1);
    free(imageOut);
    free(imageOut2);

    return 0;
}




