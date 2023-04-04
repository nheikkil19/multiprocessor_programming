__kernel void grayscaleImage(
    __read_only image2d_t imageIn,
    __write_only image2d_t imageOut,
    unsigned w,
    unsigned h,
    unsigned subpixels
) {
    for (int i=0; i<h; i++) {
        for (int j=0; j<w; j++) {
            uint4 pixel = read_imageui(imageIn, (int2)(j, i));
            uint gray = (0.299 * pixel.x + 0.587 * pixel.y + 0.114 * pixel.z);
            write_imageui(imageOut, (int2)(j, i), (uint4)(gray, 0, 0, 0));
        }
    }
}
