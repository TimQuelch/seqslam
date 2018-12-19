kernel void enhanceDiffMx(global float const* diffMx,
                          int windowSize,
                          unsigned int nPixPerThread,
                          global float* diffMxOut,
                          local float* diffVec) {
    const int offset = floor(windowSize / 2.0);
    const int rBase = nPixPerThread * get_global_id(0);
    const int nRef = get_global_size(0);
    const int q = get_global_id(1);

    for (int i = 0; i < nPixPerThread; ++i) {
        diffVec[rBase + i] = diffMx[q * nRef + rBase + i];
    }

    const int start = max(min(rBase - offset, 0),
                          min(max(rBase - offset, 0), nRef - windowSize - 1));

    const int end = min(rBase - offset + windowSize + (int)nPixPerThread, nRef);

    float sum = 0;
    for (int i = start; i < end; i++) {
        sum += diffVec[i];
    }

    for (int i = 0; i < nPixPerThread; i++) {
        const int tShift = (rBase + i - offset) < 0 ? (rBase + i - offset) : 0;
        const int bShift = (rBase - offset + windowSize + i - nRef) > 0
                               ? (rBase - offset + windowSize + i - nRef)
                               : 0;
        float mean = sum;
        for (int j = 0; j < i + tShift; j++) {
            mean -= diffVec[start + j];
        }
        for (int j = windowSize + i + bShift;
             j < windowSize + nPixPerThread - 1;
             j++) {
            mean -= diffVec[start + j];
        }
        mean /= windowSize;

        float std = 0;
        for (int j = 0; j < windowSize; j++) {
            std += pown(diffVec[start + i + j] - mean, 2);
        }

        diffMxOut[q * nRef + rBase + i] =
            (diffVec[rBase + i] - mean) / max(std / windowSize, FLT_MIN);
    }
}
