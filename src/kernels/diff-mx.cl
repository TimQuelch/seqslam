#define WARP_SIZE 32u

// Reduce a single warp
void warpReduce(local volatile float* diffs,
                unsigned int tileSizeR,
                unsigned int tileSizeQ,
                unsigned int offset,
                unsigned int pi) {
    for (unsigned int i = 0; i < tileSizeR; i++) {
        for (unsigned int j = 0; j < tileSizeQ; j++) {
            const unsigned int imageIndex = (i + j * tileSizeR) * offset;
            diffs[imageIndex + pi] += diffs[imageIndex + pi + WARP_SIZE];
            diffs[imageIndex + pi] += diffs[imageIndex + pi + (WARP_SIZE >> 1)];
            diffs[imageIndex + pi] += diffs[imageIndex + pi + (WARP_SIZE >> 2)];
            diffs[imageIndex + pi] += diffs[imageIndex + pi + (WARP_SIZE >> 3)];
            diffs[imageIndex + pi] += diffs[imageIndex + pi + (WARP_SIZE >> 4)];
            diffs[imageIndex + pi] += diffs[imageIndex + pi + (WARP_SIZE >> 5)];
        }
    }
}

kernel void diffMxNDiffs(global float const* query,
                         global float const* reference,
                         unsigned int tileSizeR,
                         unsigned int tileSizeQ,
                         unsigned int nPerThread,
                         global float* diffMxOutput,
                         local float* diffs) {
    const unsigned tr = get_group_id(1);
    const unsigned tq = get_group_id(2);
    const unsigned pi = get_local_id(0);
    const unsigned offset = get_local_size(0);
    const unsigned nPix = offset * nPerThread;

    // Load initial sums of differences into local memory
    for (unsigned q = 0; q < tileSizeQ; q++) {
        for (unsigned r = 0; r < tileSizeR; r++) {
            const unsigned imageIndex = (r + q * tileSizeR) * offset;
            diffs[pi + imageIndex] = 0; // Initialise sum to 0

            // Sum as many differences as specifed
            for (unsigned n = 0; n < nPerThread; n++) {
                diffs[pi + imageIndex] += fabs(
                    query[(tq * tileSizeQ + q) * nPix + pi + n * offset] -
                    reference[(tr * tileSizeR + r) * nPix + pi + n * offset]);
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE); // Sync threads

    // Perform parallel reduction until down to a single warp
    for (unsigned int s = get_local_size(0) / 2; s > WARP_SIZE; s >>= 1) {
        if (pi < s) {
            for (unsigned int q = 0; q < tileSizeQ; q++) {
                for (unsigned int r = 0; r < tileSizeR; r++) {
                    const unsigned int imageIndex =
                        (r + q * tileSizeR) * offset;
                    diffs[imageIndex + pi] += diffs[imageIndex + pi + s];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE); // Sync threads each iteration
    }

    // Reduce the final warp
    if (pi < WARP_SIZE) {
        warpReduce(diffs, tileSizeR, tileSizeQ, offset, pi);
    }

    // Only sync if there are more values than are in a warp
    if (tileSizeR * tileSizeQ > WARP_SIZE) {
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Save SAD the values to global memory in parallel
    if (pi < tileSizeR * tileSizeQ) {
        const unsigned int r = pi % tileSizeR;
        const unsigned int q = pi / tileSizeR;
        const unsigned int imageIndex = (r + q * tileSizeR) * offset;
        diffMxOutput[tr * tileSizeR + r +
                     (tq * tileSizeQ + q) * get_global_size(1) * tileSizeR] =
            diffs[imageIndex];
    }
}

kernel void diffMxUnrolledWarpReduce(global float const* query,
                                     global float const* reference,
                                     unsigned int tileSizeR,
                                     unsigned int tileSizeQ,
                                     unsigned int nPerThread, // Ignored
                                     global float* diffMxOutput,
                                     local float* diffs) {
    const unsigned tr = get_group_id(1);
    const unsigned tq = get_group_id(2);
    const unsigned pi = get_local_id(0);
    const unsigned offset = get_local_size(0);
    const unsigned nPix = offset * 2;

    // Load the initial differences into local memory
    for (unsigned q = 0; q < tileSizeQ; q++) {
        for (unsigned r = 0; r < tileSizeR; r++) {
            const unsigned imageIndex = (r + q * tileSizeR) * offset;
            diffs[pi + imageIndex] =
                fabs(query[(tq * tileSizeQ + q) * nPix + pi] -
                     reference[(tr * tileSizeR + r) * nPix + pi]) +
                fabs(query[(tq * tileSizeQ + q) * nPix + pi + offset] -
                     reference[(tr * tileSizeR + r) * nPix + pi + offset]);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE); // Sync threads

    // Perform parallel reduction until down to a single warp
    for (unsigned int s = get_local_size(0) / 2; s > WARP_SIZE; s >>= 1) {
        if (pi < s) {
            for (unsigned int q = 0; q < tileSizeQ; q++) {
                for (unsigned int r = 0; r < tileSizeR; r++) {
                    const unsigned int imageIndex =
                        (r + q * tileSizeR) * offset;
                    diffs[imageIndex + pi] += diffs[imageIndex + pi + s];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE); // Sync threads each iteration
    }

    // Reduce the final warp
    if (pi < WARP_SIZE) {
        warpReduce(diffs, tileSizeR, tileSizeQ, offset, pi);
    }

    // Only sync if there are more values than are in a warp
    if (tileSizeR * tileSizeQ > WARP_SIZE) {
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Save SAD the values to global memory in parallel
    if (pi < tileSizeR * tileSizeQ) {
        const unsigned int r = pi % tileSizeR;
        const unsigned int q = pi / tileSizeR;
        const unsigned int imageIndex = (r + q * tileSizeR) * offset;
        diffMxOutput[tr * tileSizeR + r +
                     (tq * tileSizeQ + q) * get_global_size(1) * tileSizeR] =
            diffs[imageIndex];
    }
}

kernel void diffMxTwoDiffs(global float const* query,
                           global float const* reference,
                           unsigned int tileSizeR,
                           unsigned int tileSizeQ,
                           unsigned int nPerThread, // Ignored
                           global float* diffMxOutput,
                           local float* diffs) {
    const unsigned tr = get_group_id(1);
    const unsigned tq = get_group_id(2);
    const unsigned pi = get_local_id(0);
    const unsigned offset = get_local_size(0);

    // Number of pixels is twice the offset as 2 pixels are processed per thread
    const unsigned nPix = offset * 2;

    // Load initial differences into local memory
    // Two separate pixels are loaded and the difference summed
    // Essentially performing the first iteration of the reduction here
    for (unsigned q = 0; q < tileSizeQ; q++) {
        for (unsigned r = 0; r < tileSizeR; r++) {
            const unsigned imageIndex = (r + q * tileSizeR) * offset;
            diffs[pi + imageIndex] =
                fabs(query[(tq * tileSizeQ + q) * nPix + pi] -
                     reference[(tr * tileSizeR + r) * nPix + pi]) +
                fabs(query[(tq * tileSizeQ + q) * nPix + pi + offset] -
                     reference[(tr * tileSizeR + r) * nPix + pi + offset]);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE); // Sync threads

    // Perform parallel reduction with continuous indexing
    for (unsigned int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (pi < s) {
            for (unsigned int q = 0; q < tileSizeQ; q++) {
                for (unsigned int r = 0; r < tileSizeR; r++) {
                    const unsigned int imageIndex =
                        (r + q * tileSizeR) * offset;
                    diffs[imageIndex + pi] += diffs[imageIndex + pi + s];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE); // Sync threads each iteration
    }

    // Save SAD the values to global memory in parallel
    if (pi < tileSizeR * tileSizeQ) {
        const unsigned int r = pi % tileSizeR;
        const unsigned int q = pi / tileSizeR;
        const unsigned int imageIndex = (r + q * tileSizeR) * offset;
        diffMxOutput[tr * tileSizeR + r +
                     (tq * tileSizeQ + q) * get_global_size(1) * tileSizeR] =
            diffs[imageIndex];
    }
}

kernel void diffMxContinuousIndex(global float const* query,
                                  global float const* reference,
                                  unsigned int tileSizeR,
                                  unsigned int tileSizeQ,
                                  unsigned int nPerThread, // Ignored
                                  global float* diffMxOutput,
                                  local float* diffs) {
    const size_t tr = get_group_id(1);
    const size_t tq = get_group_id(2);
    const size_t pi = get_local_id(0);
    const size_t nPix = get_global_size(0);

    // Load initial differences into local memory
    for (unsigned q = 0; q < tileSizeQ; q++) {
        for (unsigned r = 0; r < tileSizeR; r++) {
            const unsigned imageIndex = (r + q * tileSizeR) * nPix;
            diffs[pi + imageIndex] =
                fabs(query[(tq * tileSizeQ + q) * nPix + pi] -
                     reference[(tr * tileSizeR + r) * nPix + pi]);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE); // Sync threads

    // Perform parallel reduction with continuous indexing
    for (unsigned int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (pi < s) {
            for (unsigned int q = 0; q < tileSizeQ; q++) {
                for (unsigned int r = 0; r < tileSizeR; r++) {
                    const unsigned int imageIndex =
                        (r + q * tileSizeR) * nPix;
                    diffs[imageIndex + pi] += diffs[imageIndex + pi + s];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE); // Sync threads each iteration
    }

    // Save all SAD values to global memory, one per thread
    if (pi < tileSizeR * tileSizeQ) {
        const unsigned int r = pi % tileSizeR;
        const unsigned int q = pi / tileSizeR;
        const unsigned int imageIndex = (r + q * tileSizeR) * nPix;
        diffMxOutput[tr * tileSizeR + r +
                     (tq * tileSizeQ + q) * get_global_size(1) * tileSizeR] =
            diffs[imageIndex];
    }
}

kernel void diffMxParallelSave(global float const* query,
                               global float const* reference,
                               unsigned int tileSizeR,
                               unsigned int tileSizeQ,
                               unsigned int nPerThread, // Ignored
                               global float* diffMxOutput,
                               local float* diffs) {
    const size_t tr = get_group_id(1);
    const size_t tq = get_group_id(2);
    const size_t pi = get_local_id(0);
    const size_t nPix = get_global_size(0);

    // Load initial differences into local memory
    for (unsigned q = 0; q < tileSizeQ; q++) {
        for (unsigned r = 0; r < tileSizeR; r++) {
            const unsigned imageIndex = (r + q * tileSizeR) * nPix;
            diffs[pi + imageIndex] =
                fabs(query[(tq * tileSizeQ + q) * nPix + pi] -
                     reference[(tr * tileSizeR + r) * nPix + pi]);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE); // Sync threads

    // Perform parallel reduction
    for (unsigned int s = 1; s < get_local_size(0); s *= 2) {
        const unsigned int ti = 2 * s * pi;

        if (ti < get_local_size(0)) {
            for (unsigned int q = 0; q < tileSizeQ; q++) {
                for (unsigned int r = 0; r < tileSizeR; r++) {
                    const unsigned int imageIndex = (r + q * tileSizeR) * nPix;
                    diffs[imageIndex + ti] += diffs[imageIndex + ti + s];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE); // Sync threads after each iteration
    }

    // Save all SAD values to global memory, one per thread
    if (pi < tileSizeR * tileSizeQ) {
        const unsigned int r = pi % tileSizeR;
        const unsigned int q = pi / tileSizeR;
        const unsigned int imageIndex = (r + q * tileSizeR) * nPix;
        diffMxOutput[tr * tileSizeR + r +
                     (tq * tileSizeQ + q) * get_global_size(1) * tileSizeR] =
            diffs[imageIndex];
    }
}

kernel void diffMxNaive(global float const* query,
                        global float const* reference,
                        unsigned int tileSizeR,
                        unsigned int tileSizeQ,
                        unsigned int nPerThread, // Ignored
                        global float* diffMxOutput,
                        local float* diffs) {
    const size_t tr = get_group_id(1);
    const size_t tq = get_group_id(2);
    const size_t pi = get_local_id(0);
    const size_t nPix = get_global_size(0);

    // Load initial differences into local memory
    for (unsigned q = 0; q < tileSizeQ; q++) {
        for (unsigned r = 0; r < tileSizeR; r++) {
            const unsigned imageIndex = (r + q * tileSizeR) * nPix;
            diffs[pi + imageIndex] =
                fabs(query[(tq * tileSizeQ + q) * nPix + pi] -
                     reference[(tr * tileSizeR + r) * nPix + pi]);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE); // Sync threads

    // Perform parallel reduction
    for (unsigned int s = 1; s < get_local_size(0); s *= 2) {
        const unsigned int ti = 2 * s * pi;

        if (ti < get_local_size(0)) {
            for (unsigned int q = 0; q < tileSizeQ; q++) {
                for (unsigned int r = 0; r < tileSizeR; r++) {
                    const unsigned int imageIndex = (r + q * tileSizeR) * nPix;
                    diffs[imageIndex + ti] += diffs[imageIndex + ti + s];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE); // Sync threads after each iteration
    }

    // Save all SAD values to global memory with a single thread
    if (pi == 0) {
        for (unsigned int q = 0; q < tileSizeQ; q++) {
            for (unsigned int r = 0; r < tileSizeR; r++) {
                const unsigned int imageIndex = (r + q * tileSizeR) * nPix;
                diffMxOutput[tr * tileSizeR + r +
                             (tq * tileSizeQ + q) * get_global_size(1) *
                                 tileSizeR] = diffs[imageIndex];
            }
        }
    }
}
