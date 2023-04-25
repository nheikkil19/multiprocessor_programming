__kernel void normalizeImage(
    __read_only image2d_t imageIn,
    __write_only image2d_t imageOut,
    unsigned w,
    unsigned h
) {
    local int min, max;
    min = 255;
    max = 0;
    unsigned i = get_global_id(0);

    for (int j=0; j<w; j++) {
        uint4 pixel = read_imageui(imageIn, (int2)(j, i));
        atomic_min(&min, (int)pixel.w);
        atomic_max(&max, (int)pixel.w);
    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
    for (int j=0; j<w; j++) {
        uint4 pixel = read_imageui(imageIn, (int2)(j, i));
        int val = (int)(pixel.w - min) * 255 / (int)(max - min);
        write_imageui(imageOut, (int2)(j, i), (uint4)(val, val, val, val));
    }
}
