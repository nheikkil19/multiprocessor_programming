__kernel void calcZNCC(
    __global unsigned char *imageInL,
    __global unsigned char *imageInR,
    __global unsigned char *imageOut,
    const unsigned w,
    const unsigned h,
    const unsigned max_disp,
    const unsigned win_size,
    int inv
) {
    unsigned j = get_global_id(0);
    unsigned i = get_global_id(1);

    // Stop if outside the image
    if (i >= w || j >= h)
        return;

    float imgAvgL, imgAvgR;
    float zncc1, zncc2, zncc3, zncc, znccBest;
    unsigned char bestD;
    int x, y;
    unsigned char left, right;
    unsigned countL, countR;

    imgAvgL = 0.f;
    countL = 0;

    // Calculate sum for left window
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
            }
        }
    }
    // Calculate average
    imgAvgL = imgAvgL / countL;

    znccBest = -1;
    for (int d=0; d<=max_disp; d++) {
        // Calculate sum for right window
        imgAvgR = 0.f;
        countR = 0;
        for (int win_y=0; win_y<win_size; win_y++) {
            y = j + win_y - win_size/2;
            // Do not go outside the image
            if ( y >= 0 && y < h ) {
                for (int win_x=0; win_x<win_size; win_x++) {
                    x = i + win_x - win_size/2 - d*inv;
                    // Do not go outside the image
                    if ( x >= 0 && x < w ) {
                        imgAvgR += imageInR[y*w + x];
                        countR++;
                    }
                }
            }
        }
        // Calculate average
        imgAvgR = imgAvgR / countR;

        // Calculate ZNCC
        zncc1 = 0.f;
        zncc2 = 0.f;
        zncc3 = 0.f;
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
    // Write best value
    imageOut[j*w + i] = bestD;
}