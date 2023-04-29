__kernel void occlusionFill(
    __global unsigned char *imageIn,
    __global unsigned char *imageOut,
    unsigned w,
    unsigned h
) {
    // int win_size = get_global_size(0)s;

    __local int mid_i;
    __local int mid_j;
    __local unsigned char pixel;
    __local unsigned closest;
    __local unsigned char window[32][32];


    mid_i = get_local_size(0) / 2;
    mid_j = get_local_size(1) / 2;
    closest = 9999;

    int i = get_global_id(0);
    int j = get_global_id(1);
    int win_i = get_local_id(0);
    int win_j = get_local_id(1);

    if (i*w + j < h*w) {

        if (win_i==mid_i && win_j == mid_j) {
            pixel = imageIn[i*w + j];
        }

        if (pixel == 0) {
            window[win_i][win_j] = imageIn[i*w + j];
            int dist = max(abs(win_j-j), abs(win_i-i));
            if (window[win_i][win_j] != 0 && closest > dist) {
                atomic_min(&closest, dist);
                pixel = window[win_i][win_j];
            }
        }
    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
    if (win_i==mid_i && win_j == mid_j) {
        imageOut[i*w + j] = pixel;
    }
}
