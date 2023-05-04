__kernel void crossCheck(
    __global unsigned char *image1,
    __global unsigned char *image2,
    __global unsigned char *imageOut,
    const unsigned w,
    const unsigned h,
    const unsigned threshold
) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    // Stop if outside the image
    if (i >= h || j >= w)
        return;

    unsigned char d;
    d = image1[i*w + j];

    // Check if pixel is about the same in both images
    if ( (int)(i*w + j-d) >= 0 && abs(d - image2[i*w + j-d]) > threshold) {
        imageOut[i*w + j] = 0;
    }
    else {
        imageOut[i*w + j] = d;
    }
}
