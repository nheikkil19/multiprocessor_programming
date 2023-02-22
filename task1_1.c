#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>


int main(void) {


    // Read image. from lodepng examples
    unsigned char * image = 0;
    unsigned w, h;
    unsigned err;
    char filenameIn[] = "dataset\\im0.png";
    char filenameOut[] = "dataset\\im0_gray.png";


    err = lodepng_decode32_file(&image, &w, &h, filenameIn);
    if (err) {
        printf("Error %u\n", err);
    }
    else {
        printf("w = %u, h = %u\n", w, h);
    }

    // Resize and convert to grayscale
    unsigned hOut, wOut;
    hOut = h / 4;
    wOut = w / 4;
    unsigned char * newImage = malloc(wOut * hOut);

    for (int i=0; i<hOut; i++) {
        for (int j=0; j<wOut; j++) {
            unsigned char r, g, b;
            r = image[ (w*4)*(4 * i) + 4*j*4 + 0 ];
            g = image[ (w*4)*(4 * i) + 4*j*4 + 1 ];
            b = image[ (w*4)*(4 * i) + 4*j*4 + 2 ];
            newImage[ wOut * i + j ] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        }
    }

    // Encode to file
    LodePNGColorType colorType;
    colorType = LCT_GREY;
    err = lodepng_encode_file(filenameOut, newImage, wOut, hOut, colorType, 8);
    if (err) {
        printf("Error %u\n", err);
    }

    free(image);
    free(newImage);
    return 0;
}