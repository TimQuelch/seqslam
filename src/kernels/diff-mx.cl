kernel void diffMx(global float const* query,
                   global float const* reference,
                   unsigned width,
                   unsigned height,
                   unsigned tileSize,
                   global float* diffMxOutput) {
    const unsigned tx = get_group_id(0);
    const unsigned ty = get_group_id(1);
    const unsigned x = get_local_id(0);
    const unsigned y = get_local_id(1);

    local float diffs[width * height * tileSize * tileSize];

    for (unsigned i = 0; i < tileSize; i++) {
        for (unsigned j = 0; j < tileSize; j++) {
            const unsigned pixelIndex = x + y * width;
            const unsigned imageIndex = (i + j * tileSize) * width * height;
            diffs[pixelIndex + imageIndex] =
                fabs(query[(tx * tileSize + i) * width * height + pixelIndex] -
                     reference[(ty * tileSize + j) * width * height + pixelIndex]);
        }
    }

    // Serial reduction. Need to parallelise
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id() == 0) {
        for (unsigned i = 0; i < tileSize; i++) {
            for (unsigned j = 0; j < tileSize; j++) {
                float acc = 0;
                for (int x = 0; i < width; i++) {
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
