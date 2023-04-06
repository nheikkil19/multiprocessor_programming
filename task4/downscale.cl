__kernel void downscaleImage(
    __read_only image2d_t imageIn,
    __write_only image2d_t imageOut,
    unsigned w,
    unsigned h,
    unsigned factor,
    unsigned subpixels
) {
    unsigned hOut, wOut;
    hOut = h / factor;
    wOut = w / factor;
    unsigned i = get_global_id(0);

    for (int j=0; j<wOut; j++) {
        for (int k=0; k<subpixels; k++) {
            uint4 pixel = read_imageui(imageIn, (int2)(j*factor, i*factor));
            write_imageui(imageOut, (int2)(j, i), pixel);
        }
    }
}
