#include <stdio.h>
#include <stdlib.h>
#include "../lodepng.h"
#include <omp.h>
#include <profileapi.h>
#include <math.h>

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

    #pragma omp parallel for
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
    *imageOut = (unsigned char *) malloc(sizeof(unsigned char) * w * h);
    #pragma omp parallel for
    for (int i=0; i<h; i++) {
        unsigned r, g, b;
        for (int j=0; j<w; j++) {
            r = imageIn[i*w*subpixels + j*subpixels + 0];
            g = imageIn[i*w*subpixels + j*subpixels + 1];
            b = imageIn[i*w*subpixels + j*subpixels + 2];
            (*imageOut)[i*w + j] = (unsigned char) (0.299 * r + 0.587 * g + 0.114 * b);
        }
    }
}

void calcZNCC(unsigned char *imageL, unsigned char *imageR, unsigned char **imageOut, unsigned w, unsigned h, unsigned max_disp, unsigned win_size, int inv) {
    *imageOut = (unsigned char *) malloc(sizeof(unsigned char) * w * h);

    #pragma omp parallel for
    for (int j=0; j<h; j++) {
        double imgAvgL, imgAvgR;
        double zncc1, zncc2, zncc3, zncc;
        double znccBest, bestD;
        int x, y;
        unsigned countL, countR;

        for (int i=0; i<w; i++) {
            znccBest = -1;
            for (int d=0; d<=max_disp; d++) {
                // Calculate means over window
                imgAvgL = 0;
                imgAvgR = 0;
                countL = 0;
                countR = 0;
                for (int win_y=0; win_y<win_size; win_y++) {
                    y = j + win_y - win_size/2;
                    // Do not go outside the image
                    if ( y >= 0 && y < h ) {
                        for (int win_x=0; win_x<win_size; win_x++) {
                            x = i + win_x - win_size/2;
                            // Do not go outside the image
                            if ( x >= 0 && x < w ) {
                                // Add pixel values
                                imgAvgL += imageL[y*w + x];
                                countL++;
                            }
                            if ( x-d*inv >= 0 && x-d*inv < w ) {
                                imgAvgR += imageR[y*w + (x-d*inv)];
                                countR++;
                            }
                        }
                    }
                }
                // Calculate mean
                imgAvgL = imgAvgL / countL;
                imgAvgR = imgAvgR / countR;

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
                                int left = imageL[y*w + x];
                                int right = imageR[y*w + x-(d*inv)];
                                zncc1 += (left - imgAvgL) * (right - imgAvgR);
                                zncc2 += (left - imgAvgL) * (left - imgAvgL);
                                zncc3 += (right - imgAvgR) * (right - imgAvgR);
                            }
                        }
                    }
                }
                zncc = zncc1 / (sqrt(zncc2) * sqrt(zncc3));

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
    #pragma omp parallel for
    for (int i=0; i<w*h; i++) {
        (*imageOut)[i] = (imageIn[i] - min) * 255 / (max - min);
        }
    }


void crossCheck(unsigned char *image1, unsigned char *image2, unsigned char **imageOut, unsigned w, unsigned h, unsigned threshold) {
    *imageOut = (unsigned char *) malloc(sizeof(unsigned char) * w * h);

    #pragma omp parallel for
    for (int i=0; i<h; i++) {
        int d;
        for (int j=0; j<w; j++) {
            d = image1[i*w + j];

            if ( (int)(i*w + j-d) >= 0 && (int)abs((char)d - (char)image2[i*w + j-d]) > threshold) {
                (*imageOut)[i*w + j] = 0;
            }
            else {
                (*imageOut)[i*w + j] = (unsigned)d;
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


int main(void) {

    // Variables
    unsigned const MAX_DISP = 260/4;
    unsigned const WIN_SIZE = 9;
    unsigned const THRESHOLD = 8;
    char file1[] = "..\\dataset\\im0.png";
    char file2[] = "..\\dataset\\im1.png";
    char file3[] = "depthmap.png";
    unsigned char *image1, *image2, *imageDs1, *imageDs2, *imageGray1, *imageGray2, *imageZNCC1, *imageZNCC2, *imageCross, *imageOcc, *imageOut;
    unsigned w, h, scaleFactor;
    double start, end, startTotal, endTotal;
    double timeElapsed;

    // Start timer
    startTotal = getTime();

    // Read images
    start = getTime();
    #pragma omp parallel sections
    {
        #pragma omp section 
        readImage(file1, &image1, &w, &h);
        #pragma omp section
        readImage(file2, &image2, &w, &h);
    }
    end = getTime();
    timeElapsed = end - start;
    printf("Read files: %f s\n", timeElapsed);

    // Downscale by four
    start = getTime();
    scaleFactor = 4;
    downscaleImage(image1, &imageDs1, w, h, scaleFactor);
    downscaleImage(image2, &imageDs2, w, h, scaleFactor);
    w = w / scaleFactor;
    h = h / scaleFactor;
    end = getTime();
    timeElapsed = end - start;
    printf("Downscale: %f s\n", timeElapsed);

    // Convert to grayscale
    start = getTime();
    grayscaleImage(imageDs1, &imageGray1, w, h);
    grayscaleImage(imageDs2, &imageGray2, w, h);
    end = getTime();
    timeElapsed = end - start;
    printf("Grayscale: %f s\n", timeElapsed);

    // Do ZNCC
    start = getTime();
    calcZNCC(imageGray1, imageGray2, &imageZNCC1, w, h, MAX_DISP, WIN_SIZE, 1);
    calcZNCC(imageGray2, imageGray1, &imageZNCC2, w, h, MAX_DISP, WIN_SIZE, -1);
    end = getTime();
    timeElapsed = end - start;
    printf("ZNCC: %f s\n", timeElapsed);

    // Cross checking
    start = getTime();
    crossCheck(imageZNCC1, imageZNCC2, &imageCross, w, h, THRESHOLD);
    end = getTime();
    timeElapsed = end - start;
    printf("Cross check: %f s\n", timeElapsed);

    // Occlusion fill
    start = getTime();
    occlusionFill(imageCross, &imageOcc, w, h);
    end = getTime();
    timeElapsed = end - start;
    printf("Occlusion fill: %f s\n", timeElapsed);

    // Normalize image
    start = getTime();
    normalizeImage(imageOcc, &imageOut, w, h);
    end = getTime();
    timeElapsed = end - start;
    printf("Normalization: %f s\n", timeElapsed);

    // Save image
    start = getTime();
    writeImage(file3, imageOut, w, h);
    end = getTime();
    timeElapsed = end - start;
    printf("Save file: %f s\n", timeElapsed);

    // Free memory
    free(image1);
    free(image2);
    free(imageDs1);
    free(imageDs2);
    free(imageGray1);
    free(imageGray2);
    free(imageZNCC1);
    free(imageZNCC2);
    free(imageCross);
    free(imageOcc);
    free(imageOut);

    // Report total time
    endTotal = getTime();
    timeElapsed = endTotal - startTotal;
    printf("Total time: %f s\n", timeElapsed);

    return 0;
}




