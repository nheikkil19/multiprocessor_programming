__kernel void occlusionFill(
    __global unsigned char *imageIn,
    __global unsigned char *imageOut,
    unsigned w,
    unsigned h
) {
    const int MAX_DIST = 1000;

    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i >= h || j >= w)
        return;

    unsigned char pixel;
    if (imageIn[i*w + j] == 0) {
        // Find the closest
        pixel = 0;
        for (int d=1; d<=MAX_DIST && !pixel; d++) {
            for (int y_offset=-d+1; y_offset<=d-1 && !pixel; y_offset++) {
                if (i+y_offset>=0 && i+y_offset<h) {
                    if (j-d>=0) {
                        pixel = imageIn[(i+y_offset)*w + j-d];
                    }
                    else if (j+d<w) {
                        pixel = imageIn[(i+y_offset)*w + j+d];
                    }
                }
            }
            for (int x_offset=-d; x_offset<=d && !pixel; x_offset++) {
                if (j+x_offset>=0 &&j+x_offset<w) {
                    if (i-d>=0) {
                        pixel = imageIn[(i-d)*w + j+x_offset];
                    }
                    else if (i+d<h) {
                        pixel = imageIn[(i+d)*w + j+x_offset];
                    }
                }
            }
        }
        imageOut[i*w + j] = pixel;
    }
    else {
        imageOut[i*w + j] = imageIn[i*w + j];
    }
}
