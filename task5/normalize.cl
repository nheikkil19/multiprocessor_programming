__kernel void normalizeImage(
    __global unsigned char *imageIn,
    __global unsigned char *imageOut,
    unsigned w,
    unsigned h,
    __local float *localMem
) {
    __local int min, max;
    unsigned j = get_global_id(0);

    localMem[0*w + j] = 0;
    localMem[1*w + j] = 255;

    for (int i=0; i<h; i++) {
        int pixel = imageIn[i*w + j];
        localMem[0*w + j] = (int)fmax((float)pixel, localMem[0*w + j]);
        localMem[1*w + j] = (int)fmin((float)pixel, localMem[1*w + j]);

    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
    if (j == 0) {
        max = 0;
        for (int i=0; i<w; i++) {
            max = (int)fmax(max, localMem[0*w + i]);
        }
    }
    else if (j == 1) {
        min = 255;
        for (int i=0; i<w; i++) {
            min = (int)fmin(min, localMem[1*w + i]);
        }
    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
    for (int i=0; i<h; i++) {
        int pixel = imageIn[i*w + j];
        int val = (pixel - min) * 255 / (max - min);
        imageOut[i*w + j] = val;
    }
    if (j == 0)
        printf("min: %d, max: %d\n", min, max);
}
