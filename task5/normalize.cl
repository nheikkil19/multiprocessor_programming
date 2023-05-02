__kernel void normalizeImage(
    __global unsigned char *imageIn,
    __global unsigned char *imageOut,
    unsigned w,
    unsigned h,
    __local float *localMem
) {
    __local int min, max;
    min = 255;
    max = 0;
    unsigned j = get_global_id(0);

    for (int i=0; i<h; i++) {
        int pixel = imageIn[i*w + j];
        localMem[0*h + j] = fmax((float)pixel, localMem[1*w + j]);
        localMem[1*h + j] = fmin((float)pixel, localMem[1*w + j]);

    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
    if (j == 0) {
        for (int i=0; i<w; i++) {
            max = fmax(max, localMem[0*h + i]);
        }
    }
    else if (j == 1) {
        for (int i=0; i<w; i++) {
            min = fmin(min, localMem[1*h + i]);
        }
    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
    for (int i=0; i<h; i++) {
        int pixel = imageIn[i*w + j];
        int val = (pixel - min) * 255 / (max - min);
        imageOut[i*w + j] = val;
    }
}
