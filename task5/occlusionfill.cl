__kernel void occlusionFill(
    __read_only image2d_t imageIn,
    __write_only image2d_t imageOut,
    unsigned w,
    unsigned h
) {
    const int MAX_DIST = 1000;

    unsigned i = get_global_id(0);
    unsigned stop;
    for (int j=0; j<w; j++) {
        if (read_imageui(imageIn, (int2)(j, i)).w == 0) {
            // Find the closest
            stop = 0;
            for (int d=1; d<=MAX_DIST && !stop; d++) {
                for (int y=-d+1; y<=d-1 && !stop; y++) {
                    if (i+y>=0 && i+y<h) {
                        if (j-d>=0 && read_imageui(imageIn, (int2)(j-d, i+y)).w != 0) {
                            write_imageui(imageOut, (int2)(j, i), read_imageui(imageIn, (int2)(j-d, i+y)));
                            stop = 1;
                        }
                        else if (j+d<w && read_imageui(imageIn, (int2)(j+d, i+y)).w != 0) {
                            write_imageui(imageOut, (int2)(j, i), read_imageui(imageIn, (int2)(j+d, i+y)));
                            stop = 1;
                        }
                    }
                }
                for (int x=-d; x<=d && !stop; x++) {
                    if (j+x>=0 &&j+x<w) {
                        if (i-d>=0 && read_imageui(imageIn, (int2)(j+x, i-d)).w != 0) {
                            write_imageui(imageOut, (int2)(j, i), read_imageui(imageIn, (int2)(j+x, i-d)));
                            stop = 1;
                        }
                        else if (i+d<h && read_imageui(imageIn, (int2)(j+x, i+d)).w != 0 ) {
                            write_imageui(imageOut, (int2)(j, i), read_imageui(imageIn, (int2)(j+x, i+d)));
                            stop = 1;
                        }
                    }
                }
            }
        }
        else {
            write_imageui(imageOut, (int2)(j, i), read_imageui(imageIn, (int2)(j, i)));
        }
    }
}
