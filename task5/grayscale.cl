__kernel void grayscaleImage(
    __read_only image2d_t imageIn,
    __global unsigned char *imageOut,
    unsigned w
) {
    unsigned i = get_global_id(0);
    unsigned j = get_global_id(1);
    float4 pixel = read_imagef(imageIn, (int2)(j, i));
    float4 factors = (float4) (0.299, 0.587, 0.114, 0);
    imageOut[i*w + j] = dot(pixel, factors);
}
