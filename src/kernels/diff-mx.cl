#define WARP_SIZE 32u

void warpReduce(local volatile float* diffs,
                unsigned int tileSize,
                unsigned int offset,
                unsigned int pi) {
    for (unsigned int i = 0; i < tileSize; i++) {
        for (unsigned int j = 0; j < tileSize; j++) {
            const unsigned int imageIndex = (i + j * tileSize) * offset;
            diffs[imageIndex + pi] += diffs[imageIndex + pi + WARP_SIZE];
            diffs[imageIndex + pi] += diffs[imageIndex + pi + (WARP_SIZE >> 1)];
            diffs[imageIndex + pi] += diffs[imageIndex + pi + (WARP_SIZE >> 2)];
            diffs[imageIndex + pi] += diffs[imageIndex + pi + (WARP_SIZE >> 3)];
            diffs[imageIndex + pi] += diffs[imageIndex + pi + (WARP_SIZE >> 4)];
            diffs[imageIndex + pi] += diffs[imageIndex + pi + (WARP_SIZE >> 5)];
        }
    }
}

kernel void diffMx(global float const* query,
                   global float const* reference,
                   unsigned int tileSize,
                   unsigned int nPerThread,
                   global float* diffMxOutput,
                   local float* diffs) {
    const unsigned tx = get_group_id(1);
    const unsigned ty = get_group_id(2);
    const unsigned pi = get_local_id(0);
    const unsigned offset = get_local_size(0);
    const unsigned nPix = offset * nPerThread;

    for (unsigned i = 0; i < tileSize; i++) {
        for (unsigned j = 0; j < tileSize; j++) {
            const unsigned imageIndex = (i + j * tileSize) * offset;
            diffs[pi + imageIndex] = 0;
            for (unsigned n = 0; n < nPerThread; n++) {
                diffs[pi + imageIndex] +=
                    fabs(query[(tx * tileSize + i) * nPix + pi + n * offset] -
                         reference[(ty * tileSize + j) * nPix + pi + n * offset]);
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    for (unsigned int s = get_local_size(0) / 2; s > WARP_SIZE; s >>= 1) {
        if (pi < s) {
            for (unsigned int i = 0; i < tileSize; i++) {
                for (unsigned int j = 0; j < tileSize; j++) {
                    const unsigned int imageIndex = (i + j * tileSize) * offset;
                    diffs[imageIndex + pi] += diffs[imageIndex + pi + s];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (pi < WARP_SIZE) {
        warpReduce(diffs, tileSize, offset, pi);
    }

    if (tileSize * tileSize > WARP_SIZE) {
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (pi < tileSize * tileSize) {
        const unsigned int i = pi % tileSize;
        const unsigned int j = pi / tileSize;
        const unsigned int imageIndex = (i + j * tileSize) * offset;
        diffMxOutput[tx * tileSize + i + (ty * tileSize + j) * get_global_size(1) * tileSize] =
            diffs[imageIndex];
    }
}

kernel void diffMxNoUnroll(global float const* query,
                           global float const* reference,
                           unsigned int tileSize,
                           global float* diffMxOutput,
                           local float* diffs) {
    const unsigned tx = get_group_id(1);
    const unsigned ty = get_group_id(2);
    const unsigned pi = get_local_id(0);
    const unsigned offset = get_local_size(0);
    const unsigned nPix = offset * 2;

    for (unsigned i = 0; i < tileSize; i++) {
        for (unsigned j = 0; j < tileSize; j++) {
            const unsigned imageIndex = (i + j * tileSize) * offset;
            diffs[pi + imageIndex] = fabs(query[(tx * tileSize + i) * nPix + pi] -
                                          reference[(ty * tileSize + j) * nPix + pi]) +
                                     fabs(query[(tx * tileSize + i) * nPix + pi + offset] -
                                          reference[(ty * tileSize + j) * nPix + pi + offset]);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    for (unsigned int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (pi < s) {
            for (unsigned int i = 0; i < tileSize; i++) {
                for (unsigned int j = 0; j < tileSize; j++) {
                    const unsigned int imageIndex = (i + j * tileSize) * offset;
                    diffs[imageIndex + pi] += diffs[imageIndex + pi + s];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (pi < tileSize * tileSize) {
        const unsigned int i = pi % tileSize;
        const unsigned int j = pi / tileSize;
        const unsigned int imageIndex = (i + j * tileSize) * offset;
        diffMxOutput[tx * tileSize + i + (ty * tileSize + j) * get_global_size(1) * tileSize] =
            diffs[imageIndex];
    }
}

kernel void diffMxSingleDiff(global float const* query,
                             global float const* reference,
                             unsigned int tileSize,
                             global float* diffMxOutput,
                             local float* diffs) {
    const size_t tx = get_group_id(1);
    const size_t ty = get_group_id(2);
    const size_t pi = get_local_id(0);
    const size_t nPix = get_global_size(0);

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
                               unsigned int tileSize,
                               global float* diffMxOutput,
                               local float* diffs) {
    const size_t tx = get_group_id(1);
    const size_t ty = get_group_id(2);
    const size_t pi = get_local_id(0);
    const size_t nPix = get_global_size(0);

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
                             unsigned int tileSize,
                             global float* diffMxOutput,
                             local float* diffs) {
    const size_t tx = get_group_id(1);
    const size_t ty = get_group_id(2);
    const size_t pi = get_local_id(0);
    const size_t nPix = get_global_size(0);

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
