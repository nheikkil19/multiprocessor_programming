__kernel void calcZNCC(
    __global unsigned char *imageInL,
    __global unsigned char *imageInR,
    __global unsigned char *imageOut,
    unsigned w,
    unsigned h,
    unsigned max_disp,
    unsigned win_size,
    int inv
) {
    unsigned j = get_global_id(0);
    unsigned i = get_global_id(1);
    double imgAvgL, imgAvgR;
    double zncc1, zncc2, zncc3, zncc;
    double znccBest, bestD;
    int x, y;
    int left, right;
    unsigned countL, countR;

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
                        imgAvgL += imageInL[y*w + x];
                        countL++;
                    }
                    if ( x-d*inv >= 0 && x-d*inv < w ) {
                        imgAvgR += imageInR[y*w + x-d*inv];
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
                        left = imageInL[y*w + x];
                        right = imageInR[y*w + x-d*inv];
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
    imageOut[j*w + i] = bestD;
}