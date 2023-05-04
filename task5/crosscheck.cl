__kernel void crossCheck(
    __global unsigned char *image1,
    __global unsigned char *image2,
    __global unsigned char *imageOut,
    unsigned w,
    unsigned h,
    unsigned threshold
) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int d;
    d = image1[i*w + j];

    if (i > h || j > w) return;

    if ( (int)(i*w + j-d) >= 0 && abs(d - image2[i*w + j-d]) > threshold) {
        imageOut[i*w + j] = 0;
    }
    else {
        imageOut[i*w + j] = d;
    }
}
