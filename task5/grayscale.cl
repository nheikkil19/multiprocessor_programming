__kernel void grayscaleImage(
    __read_only image2d_t imageIn,
    __global unsigned char *imageOut
) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    int h = get_image_height(imageIn);
    int w = get_image_width(imageIn);

    // Stop if outside the image
    if (i >= h || j >= w)
        return;

    // Read the pixel
    int4 pixel4 = read_imagei(imageIn, (int2)(j, i));
    // Select the first 3 values
    int3 pixel3 = (int3) pixel4.s123;
    // Define factors for conversion
    float3 factors = (float3) (0.299, 0.587, 0.114);
    // Compute the gray value
    imageOut[i*w + j] = dot(convert_float3(pixel3), factors);
}
