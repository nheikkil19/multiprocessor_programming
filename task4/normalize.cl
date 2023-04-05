__kernel void normalizeImage(
    __read_only image2d_t imageIn,
    __write_only image2d_t imageOut,
    unsigned w,
    unsigned h
) {
    local unsigned min, max;
    min = 255;
    max = 0;
    unsigned i = get_global_id(0);

    for (int j=0; j<w; j++) {
        uint4 pixel = read_imageui(imageIn, (int2)(j, i));
        if (pixel.w < min) {
            min = pixel.w;
        }
        if (pixel.w > max) {
            max = pixel.w;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int j=0; j<w; j++) {
        uint4 pixel = read_imageui(imageIn, (int2)(j, i));
        unsigned val = (pixel.w - min) * 255 / (max - min);
        write_imageui(imageOut, (int2)(j, i), (uint4)(val, val, val, val));
    }
}
