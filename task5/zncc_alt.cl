__kernel void calcZNCC(
    __global unsigned char *imageInL,
    __global unsigned char *imageInR,
    __global unsigned char *imageOut,
    __local unsigned char *windowL,
    __local unsigned char *windowR,
    unsigned w,
    unsigned h,
    unsigned max_disp,
    unsigned win_size,
    int inv
) {
    unsigned gid_y = get_global_id(0);
    unsigned gid_x = get_global_id(1);

    unsigned lid_y = get_local_id(0);
    unsigned lid_x = get_local_id(1);

    int mid_y, mid_x;

    __local int imgAvgL, imgAvgR;
    __local int zncc1, zncc2, zncc3, zncc;
    __local int znccBest, bestD;
    int coordL, coordR;
    int left, right;
    __local unsigned countL, countR;
    countL = 9;
    countR = 9;

    znccBest = -1;
    for (int d=0; d<=max_disp; d++) {
        // Cache windows
        for (int offset_y=0; offset_y<win_size; offset_y++) {
            for (int offset_x=0; offset_x<win_size; offset_x++) {
                mid_y = gid_y * win_size + offset_y;
                mid_x = gid_x * win_size + offset_x;
                // TODO: check borders -> get countL, countR
                coordL = (mid_y-win_size/2+lid_y)*w + (mid_x-win_size/2+lid_x);
                coordR = (mid_y-win_size/2+lid_y)*w + (mid_x-win_size/2+lid_x) - d*inv;
                if (coordL < 0 || coordL >= w*h) {
                    windowL[lid_y*win_size + lid_x] = 0;
                    countL--;
                }
                else {
                    windowL[lid_y*win_size + lid_x] = imageInL[coordL];
                }
                if (coordR < 0 || coordR >= w*h) {
                    windowR[lid_y*win_size + lid_x] = 0;
                    countR--;
                }
                else {
                    windowR[lid_y*win_size + lid_x] = imageInR[coordR];
                }
            }
        }

        // Calculate means over window
        imgAvgL = 0;
        imgAvgR = 0;
        countL = 0;
        countR = 0;

        // // Calculate average of pixel values over window.
        atomic_add(&imgAvgL, windowL[lid_y*win_size + lid_x]);
        atomic_add(&imgAvgR, windowR[lid_y*win_size + lid_x]);

        // Calculate mean
        imgAvgL = imgAvgL / countL;
        imgAvgR = imgAvgR / countR;

        // Calculate ZNCC
        zncc1 = 0;
        zncc2 = 0;
        zncc3 = 0;

        // TODO: Check borders
        left = imageInL[lid_y*w + lid_x];
        right = imageInR[lid_y*w + lid_x];
        if (right != 0) {
            atomic_add(&zncc1, (left - imgAvgL) * (right - imgAvgR));
            atomic_add(&zncc2, (left - imgAvgL) * (left - imgAvgL));
            atomic_add(&zncc3, (right - imgAvgR) * (right - imgAvgR));
        }
        work_group_barrier(CLK_LOCAL_MEM_FENCE);
        zncc = zncc1 / (sqrt((double)zncc2) * sqrt((double)zncc3));

        // Select if better than current best
        if (zncc > znccBest) {
            znccBest = zncc;
            bestD = d;
        }
    }
    if (mid_y*w + mid_x < w*h) {
        imageOut[mid_y*w + mid_x] = bestD;
    }
}
