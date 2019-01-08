kernel void enhanceDiffMx(global float const* diffMx,
                          int windowSize,
                          unsigned int nPixPerThread,
                          global float* diffMxOut,
                          local float* diffVec) {
    const int offset = floor(windowSize / 2.0);
    const int rBase = nPixPerThread * get_global_id(0);
    const int nRef = get_global_size(0);
    const int nQue = get_global_size(1);
    const int q = get_global_id(1);

    for (int i = 0; i < nPixPerThread; ++i) {
        diffVec[rBase + i] = diffMx[(rBase + i) * nQue + q];
    }
    barrier(CLK_LOCAL_MEM_FENCE); // Sync threads

    float sum = 0;
    for (int i = rBase; i < rBase + windowSize + nPixPerThread; i++) {
        if (i >= 0 && i < nRef) {
            sum += diffVec[i];
        }
    }

    for (int i = 0; i < nPixPerThread; i++) {

        const int topCut = -min(rBase - offset + i, 0);
        const int botCut = max(rBase - offset + windowSize + i, nRef) - nRef;
        const int cut = topCut + botCut;

        float mean = sum;
        for (int j = rBase - offset; j < rBase - offset + windowSize + i; j++) {
            if (j >= 0 && j < nRef) {
                mean -= diffVec[j];
            }
        }
        for (int j = rBase - offset + windowSize + i;
             j <= rBase - offset + windowSize + nPixPerThread;
             j++) {
            if (j >= 0 && j < nRef) {
                mean -= diffVec[j];
            }
        }
        mean /= (windowSize - cut);

        float std = 0;
        for (int j = rBase - offset + i; j < rBase - offset + windowSize + i;
             j++) {
            if (j >= 0 && j < nRef) {
                std += pown(diffVec[j] - mean, 2);
            }
        }

        diffMxOut[(rBase + i) * nQue + q] =
            (diffVec[rBase + i] - mean) /
            max(std / (windowSize - cut), FLT_MIN);
    }
}