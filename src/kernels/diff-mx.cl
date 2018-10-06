kernel void diffMx(global float const* query,
                   global float const* reference,
                   unsigned int nPix,
                   unsigned int tileSize,
                   global float* diffMxOutput,
                   local float* diffs) {
    const size_t tx = get_group_id(0);
    const size_t ty = get_group_id(1);
    const size_t pi = get_local_id(2);

    for (size_t i = 0; i < tileSize; i++) {
        for (size_t j = 0; j < tileSize; j++) {
            const size_t imageIndex = (i + j * tileSize) * nPix;
            diffs[pi + imageIndex] = fabs(query[(tx * tileSize + i) * nPix + pi] -
                                          reference[(ty * tileSize + j) * nPix + pi]);
        }
    }

    // Serial reduction. Need to parallelise
    barrier(CLK_LOCAL_MEM_FENCE);
    if (pi == 0) {
        for (size_t i = 0; i < tileSize; i++) {
            for (size_t j = 0; j < tileSize; j++) {
                float acc = 0;
                for (int p = 0; p < nPix; p++) {
                    const size_t imageIndex = (i + j * tileSize) * nPix;
                    acc += diffs[p + imageIndex];
                }
                diffMxOutput[tx * tileSize + i +
                             (ty * tileSize + j) * get_global_size(0) * tileSize] = acc;
            }
        }
    }
}
