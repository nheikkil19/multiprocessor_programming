__kernel void occlusionFill(
    __global unsigned char *imageIn,
    __global unsigned char *imageOut,
    const unsigned w,
    const unsigned h
) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    const int MAX_DIST = 1000;      // Maximum distance of closest pixel
    unsigned char pixel;

    // Stop if outside the image
    if (i >= h || j >= w)
        return;

    if (imageIn[i*w + j] == 0) {
        // Find the closest
        pixel = 0;
        // Start from the closest and go outwards
        for (int d=1; d<=MAX_DIST && !pixel; d++) {
            // Look the sides of the square
            for (int y_offset=-d+1; y_offset<=d-1 && !pixel; y_offset++) {
                if (i+y_offset>=0 && i+y_offset<h) {
                    // Left
                    if (j-d>=0) {
                        pixel = imageIn[(i+y_offset)*w + j-d];
                    }
                    // Right
                    else if (j+d<w) {
                        pixel = imageIn[(i+y_offset)*w + j+d];
                    }
                }
            }
            // Look the top and bottom of the square and corners
            for (int x_offset=-d; x_offset<=d && !pixel; x_offset++) {
                if (j+x_offset>=0 &&j+x_offset<w) {
                    // Top
                    if (i-d>=0) {
                        pixel = imageIn[(i-d)*w + j+x_offset];
                    }
                    // Bottom
                    else if (i+d<h) {
                        pixel = imageIn[(i+d)*w + j+x_offset];
                    }
                }
            }
        }
        // Write the pixel that was found
        imageOut[i*w + j] = pixel;
    }
    else {
        imageOut[i*w + j] = imageIn[i*w + j];
    }
}
