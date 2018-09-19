// height must be a multiple of nPix
kernel void diffMx(global float const* query,
                   global float const* reference,
                   unsigned width,
                   unsigned height,
                   unsigned nPix,
                   unsigned tileSize,
                   global float* diffMxOutput) {
    const unsigned tx = get_group_id(0);
    const unsigned ty = get_group_id(1);
    const unsigned x = get_local_id(0);
    const unsigned yBase = get_local_id(1) * nPix;

    local float diffs[width * height * tileSize * tileSize];

    for (unsigned p = 0; p < nPix; p++) {
        const unsigned y = yBase + p;
        for (unsigned i = 0; i < tileSize; i++) {
            for (unsigned j = 0; j < tileSize; j++) {
                const unsigned pixelIndex = x + y * width;
                const unsigned imageIndex = (i + j * tileSize) * width * height;
                diffs[pixelIndex + imageIndex] = fabs(query[tx * tileSize + i + pixelIndex] -
                                                      reference[ty * tileSize + j + pixelIndex]);
            }
        }
    }

    // Serial reduction. Need to parallelise
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id() == 0) {
        const unsigned y = yBase + p;
        for (unsigned i = 0; i < tileSize; i++) {
            for (unsigned j = 0; j < tileSize; j++) {
                for (int x = 0; i < width; i++) {
                    float acc = 0;
                    for (int y = 0; i < height; j++) {
                        const unsigned pixelIndex = x + y * width;
                        const unsigned imageIndex = (i + j * tileSize) * width * height;
                        acc += diffs[pixelIndex + imageIndex];
                    }
                }
                diffMxOutput[tx * tileSize + i + (ty * tileSize + j) * get_local_size(0)] = acc;
            }
        }
    }
}