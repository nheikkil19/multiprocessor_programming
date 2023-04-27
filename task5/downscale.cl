__kernel void downscaleImage(
    __read_only image2d_t imageIn,
    __write_only image2d_t imageOut,
    unsigned factor
) {
    unsigned i = get_global_id(0);
    unsigned j = get_global_id(1);

    uint4 pixel = read_imageui(imageIn, (int2)(j*factor, i*factor));
    write_imageui(imageOut, (int2)(j, i), pixel);
}
