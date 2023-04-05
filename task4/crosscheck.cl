__kernel void crossCheck(
    __read_only image2d_t image1,
    __read_only image2d_t image2,
    __write_only image2d_t imageOut,
    unsigned w,
    unsigned h,
    unsigned threshold
) {
    unsigned i = get_global_id(0);
    // for (int i=0; i<h; i++) {
    int d;
    for (int j=0; j<w; j++) {
        d = read_imageui(image1, (int2)(j, i)).w;
        // printf("d: %d, i: %d, j: %d\n", d, i, j);

        if ( i*w + j - d >= 0 && abs(d - (int)read_imagei(image2, (int2)(j-d, i)).w) > threshold) {
            write_imageui(imageOut, (int2)(j, i), (uint4)(0, 0, 0, 0));
        }
        else {
            write_imageui(imageOut, (int2)(j, i), (uint4)(d, d, d, d));
        }
    }
    // }
}
