#include <stdio.h>
#include <stdlib.h>
#include "lodepng.h"
#include <math.h>
#include <time.h>

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
            (*imageOut)[i*w + j] = (int) (0.299 * r + 0.587 * g + 0.114 * b);
        }
    }
}

void calcZNCC(unsigned char *image1, unsigned char *image2, unsigned char **imageOut, unsigned w, unsigned h, unsigned max_disp, unsigned win_size, int inv) {
    double imgAvg1, imgAvg2;
    double zncc1, zncc2, zncc3, zncc;
    double znccBest, bestD;
    int x, y;
    unsigned count;

    *imageOut = (unsigned char *) malloc(sizeof(unsigned char) * w * h);

    for (int j=0; j<h; j++) {
        for (int i=0; i<w; i++) {
            znccBest = -1;
            for (int d=0; d<max_disp; d++) {
                // Calculate means over window
                imgAvg1 = 0;
                imgAvg2 = 0;
                count = 0;
                for (int win_y=0; win_y<win_size; win_y++) {
                    y = j + win_y - win_size/2;
                    // Do not go outside the image
                    if ( y >= 0 && y < h ) {
                        for (int win_x=0; win_x<win_size; win_x++) {
                            x = i + win_x - win_size/2;
                            // Do not go outside the image
                            if ( x >= 0 && x < w ) {
                                // Add pixel values
                                imgAvg1 += image1[y*w + x];
                                imgAvg2 += image2[y*w + x];
                                count++;
                            }
                        }
                    }
                }
                // Calculate mean
                imgAvg1 = imgAvg1 / count;
                imgAvg2 = imgAvg2 / count;

                // Calculate ZNCC
                zncc1 = 0;
                zncc2 = 0;
                zncc3 = 0;
                for (int win_y=0; win_y<win_size; win_y++) {
                    y = j + win_y - win_size/2;
                    // Do not go outside the image
                    if ( y >= 0 && y < h ) {
                        for (int win_x=0; win_x<win_size; win_x++) {
                            x = i + win_x - win_size/2;
                            // Do not go outside the image
                            if ( x >= 0 && x-(d*inv) >= 0 && x < w && x-(d*inv) < w ) {
                                zncc1 += ((double)image1[y*w + x] - imgAvg1) * ((double)image2[y*w + x-(d*inv)] - imgAvg2);
                                zncc2 += abs((double)image1[y*w + x] - imgAvg1);
                                zncc3 += abs((double)image2[y*w + x-(d*inv)] - imgAvg2);
                            }
                        }
                    }
                }
                zncc = zncc1 / (zncc2 * zncc3);

                // Select if better than current best
                if (zncc > znccBest) {
                    znccBest = zncc;
                    bestD = d;
                }
            }
            (*imageOut)[j*w + i] = bestD;
        }
    }
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
    for (int i=0; i<w*h; i++) {
        (*imageOut)[i] = (imageIn[i] - min) * 255 / (max - min);
    }
}


void crossCheck(unsigned char *image1, unsigned char *image2, unsigned char **imageOut, unsigned w, unsigned h, unsigned threshold) {
    *imageOut = (unsigned char *) malloc(sizeof(unsigned char) * w * h);
    int x, y, d;

    for (int i=0; i<h; i++) {
        for (int j=0; j<w; j++) {
            y = i * w;
            x = j;
            d = image1[y + x];

            if ( y + x >= 0 && y + (x - d) >= 0 && y + x < w*h) {
                if (abs((char)image1[y + x] - (char)image2[y + (x-d)]) > threshold) {
                    (*imageOut)[y + x] = 0;
                }
                else {
                    (*imageOut)[y + x] = image1[y + x];
                }
            }
        }
    }
}

void occlusionFill(unsigned char *imageIn, unsigned char **imageOut, unsigned w, unsigned h) {
    const int MAX_DIST = 1000;
    unsigned stop;
    *imageOut = (unsigned char *) malloc(sizeof(unsigned char) * w * h);
    for (int i=0; i<h; i++) {
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

int main(void) {

    // Variables
    unsigned const MAX_DISP = 260/4;
    unsigned const WIN_SIZE = 15;
    unsigned const THRESHOLD = 8;
    char file1[] = "dataset\\im0.png";
    char file2[] = "dataset\\im1.png";
    char file3[64] = "dataset\\depthmap.png";
    unsigned char *image1, *image2, *imageDs1, *imageDs2, *imageGray1, *imageGray2, *imageZNCC1, *imageZNCC2, *imageNorm1, *imageNorm2, *imageCross, *imageOut;
    unsigned w, h, scaleFactor;
    clock_t start, end;
    double timeElapsed;


    // Read images
    start = clock();
    readImage(file1, &image1, &w, &h);
    readImage(file2, &image2, &w, &h);
    end = clock();
    timeElapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Read files: %.3f s\n", timeElapsed);

    // Downscale by four
    start = clock();
    scaleFactor = 4;
    downscaleImage(image1, &imageDs1, w, h, scaleFactor);
    downscaleImage(image2, &imageDs2, w, h, scaleFactor);
    w = w / scaleFactor;
    h = h / scaleFactor;
    end = clock();
    timeElapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Downscale: %.3f s\n", timeElapsed);

    // Convert to grayscale
    start = clock();
    grayscaleImage(imageDs1, &imageGray1, w, h);
    grayscaleImage(imageDs2, &imageGray2, w, h);
    end = clock();
    timeElapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Grayscale: %.3f s\n", timeElapsed);

    // Do ZNCC
    start = clock();
    calcZNCC(imageGray1, imageGray2, &imageZNCC1, w, h, MAX_DISP, WIN_SIZE, 1);
    calcZNCC(imageGray2, imageGray1, &imageZNCC2, w, h, MAX_DISP, WIN_SIZE, -1);
    end = clock();
    timeElapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("ZNCC: %.3f s\n", timeElapsed);

    // Normalize image
    start = clock();
    normalizeImage(imageZNCC1, &imageNorm1, w, h);
    normalizeImage(imageZNCC2, &imageNorm2, w, h);
    end = clock();
    timeElapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Normalization: %.3f s\n", timeElapsed);

    // Cross checking
    start = clock();
    crossCheck(imageNorm1, imageNorm2, &imageCross, w, h, THRESHOLD);
    end = clock();
    timeElapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Cross check: %.3f s\n", timeElapsed);

    // Occlusion fill
    start = clock();
    occlusionFill(imageCross, &imageOut, w, h);
    end = clock();
    timeElapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Occlusion fill: %.3f s\n", timeElapsed);


    // Save image
    start = clock();
    sprintf(file3, "dataset\\depthmap_%d_%d.png", WIN_SIZE, THRESHOLD);
    writeImage(file3, imageOut, w, h);
    end = clock();
    timeElapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Save file: %.3f s\n", timeElapsed);

    // Free memory
    free(image1);
    free(imageDs1);
    free(imageDs2);
    free(imageGray1);
    free(imageGray2);
    free(imageZNCC1);
    free(imageZNCC2);
    free(imageNorm1);
    free(imageNorm2);
    free(imageCross);
    free(imageOut);

    return 0;
}




