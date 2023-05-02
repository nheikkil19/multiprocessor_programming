__kernel void normalizeImage(
    __global unsigned char *imageIn,
    __global unsigned char *imageOut,
    unsigned w,
    unsigned h
) {
    local int min, max;
    min = 255;
    max = 0;
    unsigned j = get_global_id(0);

    for (int i=0; i<h; i++) {
        int pixel = imageIn[i*w + j];
        atomic_min(&min, pixel);
        atomic_max(&max, pixel);
    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
    for (int i=0; i<h; i++) {
        int pixel = imageIn[i*w + j];
        int val = (pixel - min) * 255 / (max - min);
        imageOut[i*w + j] = val;
    }
}
