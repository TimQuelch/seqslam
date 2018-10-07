kernel void diffMx(global float const* query,
                   global float const* reference,
                   unsigned int nPix,
                   unsigned int tileSize,
                   global float* diffMxOutput,
                   local float* diffs) {
    const size_t tx = get_group_id(1);
    const size_t ty = get_group_id(2);
    const size_t pi = get_local_id(0);

    for (size_t i = 0; i < tileSize; i++) {
        for (size_t j = 0; j < tileSize; j++) {
            const size_t imageIndex = (i + j * tileSize) * nPix;
            diffs[pi + imageIndex] = fabs(query[(tx * tileSize + i) * nPix + pi] -
                                          reference[(ty * tileSize + j) * nPix + pi]);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    for (unsigned int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (pi < s) {
            for (unsigned int i = 0; i < tileSize; i++) {
                for (unsigned int j = 0; j < tileSize; j++) {
                    const unsigned int imageIndex = (i + j * tileSize) * nPix;
                    diffs[imageIndex + pi] += diffs[imageIndex + pi + s];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (pi < tileSize * tileSize) {
        const unsigned int i = pi % tileSize;
        const unsigned int j = pi / tileSize;
        const unsigned int imageIndex = (i + j * tileSize) * nPix;
        diffMxOutput[tx * tileSize + i + (ty * tileSize + j) * get_global_size(1) * tileSize] =
            diffs[imageIndex];
    }
}

kernel void diffMxSingleDiff(global float const* query,
                             global float const* reference,
                             unsigned int nPix,
                             unsigned int tileSize,
                             global float* diffMxOutput,
                             local float* diffs) {
    const size_t tx = get_group_id(1);
    const size_t ty = get_group_id(2);
    const size_t pi = get_local_id(0);

    for (size_t i = 0; i < tileSize; i++) {
        for (size_t j = 0; j < tileSize; j++) {
            const size_t imageIndex = (i + j * tileSize) * nPix;
            diffs[pi + imageIndex] = fabs(query[(tx * tileSize + i) * nPix + pi] -
                                          reference[(ty * tileSize + j) * nPix + pi]);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    for (unsigned int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (pi < s) {
            for (unsigned int i = 0; i < tileSize; i++) {
                for (unsigned int j = 0; j < tileSize; j++) {
                    const unsigned int imageIndex = (i + j * tileSize) * nPix;
                    diffs[imageIndex + pi] += diffs[imageIndex + pi + s];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (pi < tileSize * tileSize) {
        const unsigned int i = pi % tileSize;
        const unsigned int j = pi / tileSize;
        const unsigned int imageIndex = (i + j * tileSize) * nPix;
        diffMxOutput[tx * tileSize + i + (ty * tileSize + j) * get_global_size(1) * tileSize] =
            diffs[imageIndex];
    }
}

kernel void diffMxStridedIndex(global float const* query,
                               global float const* reference,
                               unsigned int nPix,
                               unsigned int tileSize,
                               global float* diffMxOutput,
                               local float* diffs) {
    const size_t tx = get_group_id(1);
    const size_t ty = get_group_id(2);
    const size_t pi = get_local_id(0);

    for (size_t i = 0; i < tileSize; i++) {
        for (size_t j = 0; j < tileSize; j++) {
            const size_t imageIndex = (i + j * tileSize) * nPix;
            diffs[pi + imageIndex] = fabs(query[(tx * tileSize + i) * nPix + pi] -
                                          reference[(ty * tileSize + j) * nPix + pi]);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    for (unsigned int s = 1; s < get_local_size(0); s *= 2) {
        const unsigned int ti = 2 * s * pi;

        if (ti < get_local_size(0)) {
            for (unsigned int i = 0; i < tileSize; i++) {
                for (unsigned int j = 0; j < tileSize; j++) {
                    const unsigned int imageIndex = (i + j * tileSize) * nPix;
                    diffs[imageIndex + ti] += diffs[imageIndex + ti + s];
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (pi < tileSize * tileSize) {
        const unsigned int i = pi % tileSize;
        const unsigned int j = pi / tileSize;
        const unsigned int imageIndex = (i + j * tileSize) * nPix;
        diffMxOutput[tx * tileSize + i + (ty * tileSize + j) * get_global_size(1) * tileSize] =
            diffs[imageIndex];
    }
}

kernel void diffMxSerialSave(global float const* query,
                             global float const* reference,
                             unsigned int nPix,
                             unsigned int tileSize,
                             global float* diffMxOutput,
                             local float* diffs) {
    const size_t tx = get_group_id(1);
    const size_t ty = get_group_id(2);
    const size_t pi = get_local_id(0);

    for (size_t i = 0; i < tileSize; i++) {
        for (size_t j = 0; j < tileSize; j++) {
            const size_t imageIndex = (i + j * tileSize) * nPix;
            diffs[pi + imageIndex] = fabs(query[(tx * tileSize + i) * nPix + pi] -
                                          reference[(ty * tileSize + j) * nPix + pi]);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    for (unsigned int s = 1; s < get_local_size(0); s *= 2) {
        const unsigned int ti = 2 * s * pi;

        if (ti < get_local_size(0)) {
            for (unsigned int i = 0; i < tileSize; i++) {
                for (unsigned int j = 0; j < tileSize; j++) {
                    const unsigned int imageIndex = (i + j * tileSize) * nPix;
                    diffs[imageIndex + ti] += diffs[imageIndex + ti + s];
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (pi == 0) {
        for (unsigned int i = 0; i < tileSize; i++) {
            for (unsigned int j = 0; j < tileSize; j++) {
                const unsigned int imageIndex = (i + j * tileSize) * nPix;
                diffMxOutput[tx * tileSize + i +
                             (ty * tileSize + j) * get_global_size(1) * tileSize] =
                    diffs[imageIndex];
            }
        }
    }
}
