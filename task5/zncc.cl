__kernel void calcZNCC(
    __read_only image2d_t imageInL,
    __read_only image2d_t imageInR,
    __write_only image2d_t imageOut,
    unsigned w,
    unsigned h,
    unsigned max_disp,
    unsigned win_size,
    int inv
) {
    unsigned j = get_global_id(0);
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
                            imgAvgL += read_imageui(imageInL, (int2)(x, y)).x;
                            countL++;
                        }
                        if ( x-d*inv >= 0 && x-d*inv < w ) {
                            imgAvgR += read_imageui(imageInR, (int2)(x-d*inv, y)).x;
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
                            int left = read_imageui(imageInL, (int2)(x, y)).x;
                            int right = read_imageui(imageInR, (int2)(x-(d*inv), y)).x;
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
        write_imageui(imageOut, (int2)(i, j), (uint4)(bestD, bestD, bestD, bestD));
    }
}