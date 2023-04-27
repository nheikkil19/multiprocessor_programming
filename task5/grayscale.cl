__kernel void grayscaleImage(
    __read_only image2d_t imageIn,
    __global unsigned char *imageOut,
    unsigned w
) {
    unsigned i = get_global_id(0);
    unsigned j = get_global_id(1);
    uint4 pixel = read_imageui(imageIn, (int2)(j, i));
    unsigned gray = (0.299 * pixel.x + 0.587 * pixel.y + 0.114 * pixel.z);
    imageOut[i*w + j] = gray;
}
