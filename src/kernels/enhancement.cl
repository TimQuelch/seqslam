kernel void enhanceDiffMx(global float const* diffMx,
                          int windowSize,
                          global float* diffMxOut,
                          local float* diffVec) {
    const int offset = floor(windowSize / 2.0);
    const int r = get_global_id(0);
    const int q = get_global_id(1);
    diffVec[r] = diffMx[q * get_global_size(0) + r];

    const int start =
        max(min(q - offset, 0),
            min(max(q - offset, 0), (int)get_global_size(0) - windowSize - 1));

    float mean = 0;
    for (int i = 0; i < windowSize; i++) {
        mean += diffVec[start + i];
    }
    mean /= windowSize;

    float std = 0;
    for (int i = 0; i < windowSize; i++) {
        std += pown(diffVec[start + i] - mean, 2);
    }

    diffMxOut[q * get_global_size(0) + r] =
        (diffVec[r] - mean) / max(std / windowSize, FLT_MIN);
}
