__kernel void grayscaleImage(
    __read_only image2d_t imageIn,
    __global unsigned char *imageOut,
    unsigned w
) {
    unsigned i = get_global_id(0);
    unsigned j = get_global_id(1);

    if (i >= get_image_height(imageIn) || j >= get_image_width(imageIn))
        return;

    int4 pixel4 = read_imagei(imageIn, (int2)(j, i));
    int3 pixel3 = (int3) (pixel4.x, pixel4.y, pixel4.z);
    float3 factors = (float3) (0.299, 0.587, 0.114);
    imageOut[i*w + j] = dot(convert_float3(pixel3), factors);
}
