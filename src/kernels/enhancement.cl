kernel void enhanceDiffMx(global float const* diffMx,
                          int windowSize,
                          unsigned int nPixPerThread,
                          global float* diffMxOut,
                          local float* diffVec) {
    const int offset = floor(windowSize / 2.0);
    const int rBase = nPixPerThread * get_global_id(0);
    const int q = get_global_id(1);

    for (int i = 0; i < nPixPerThread; ++i) {
        diffVec[rBase + i] = diffMx[q * get_global_size(0) + rBase + i];
    }

    const int start =
        max(min(q - offset, 0),
            min(max(q - offset, 0), (int)get_global_size(0) - windowSize - 1));

    float sum = 0;
    for (int i = 0; i < windowSize + nPixPerThread - 1; i++) {
        sum += diffVec[start + i];
    }

    for (int i = 0; i < nPixPerThread; i++) {
        float mean = sum;
        for (int j = 0; j < i; j++) {
            mean -= diffVec[start + j];
        }
        for (int j = windowSize + i; j < windowSize + nPixPerThread - 1; j++) {
            mean -= diffVec[start + j];
        }
        mean /= windowSize;

        float std = 0;
        for (int j = 0; j < windowSize; j++) {
            std += pown(diffVec[start + i + j] - mean, 2);
        }

        diffMxOut[q * get_global_size(0) + rBase + i] =
            (diffVec[rBase + i] - mean) / max(std / windowSize, FLT_MIN);
    }
}
