__kernel void normalizeImage(
    __global unsigned char *imageIn,
    __global unsigned char *imageOut,
    unsigned w,
    unsigned h
) {
    local int min, max;
    min = 255;
    max = 0;
    unsigned i = get_global_id(0);

    for (int j=0; j<w; j++) {
        int pixel = imageIn[i*w + j];
        atomic_min(&min, pixel);
        atomic_max(&max, pixel);
    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
    for (int j=0; j<w; j++) {
        int pixel = imageIn[i*w + j];
        int val = (pixel - min) * 255 / (max - min);
        imageOut[i*w + j] = val;
    }
}
