__kernel void occlusionFill(
    __global unsigned char *imageIn,
    __global unsigned char *imageOut,
    unsigned w,
    unsigned h
) {
    const int MAX_DIST = 1000;

    int i = get_global_id(0);
    int j = get_global_id(1);
    unsigned stop;
    if (imageIn[i*w + j] == 0) {
        // Find the closest
        stop = 0;
        for (int d=1; d<=MAX_DIST && !stop; d++) {
            for (int y=-d+1; y<=d-1 && !stop; y++) {
                if (i+y>=0 && i+y<h) {
                    if (j-d>=0 && imageIn[(i+y)*w + j-d] != 0) {
                        imageOut[i*w + j] = imageIn[(i+y)*w + j-d];
                        stop = 1;
                    }
                    else if (j+d<w && imageIn[(i+y)*w + j+d] != 0) {
                        imageOut[i*w + j] = imageIn[(i+y)*w + j+d];
                        stop = 1;
                    }
                }
            }
            for (int x=-d; x<=d && !stop; x++) {
                if (j+x>=0 &&j+x<w) {
                    if (i-d>=0 && imageIn[(i-d)*w + j+x] != 0) {
                        imageOut[i*w + j] = imageIn[(i-d)*w + j+x];
                        stop = 1;
                    }
                    else if (i+d<h && imageIn[(i+d)*w + j+x] != 0) {
                        imageOut[i*w + j] = imageIn[(i+d)*w + j+x];
                        stop = 1;
                    }
                }
            }
        }
    }
    else {
        imageOut[i*w + j] = imageIn[i*w + j];
    }
}
