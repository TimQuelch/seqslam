kernel void sequenceSearch(global float const* diffMx,
                           constant int* qOffsets,
                           constant int* rOffsets,
                           unsigned int sequenceLength,
                           unsigned int nTrajectories,
                           unsigned int nPixPerThread,
                           global float* sequenceStrengthOut) {
    const unsigned int rBase = get_global_id(0);
    const unsigned int q = get_global_id(1);
    const unsigned int nThreads = get_global_size(0);
    const unsigned int nRef = nPixPerThread * nThreads;
    const unsigned int nQue = get_global_size(1);

    const int minQOffset = qOffsets[0];
    const int maxQOffset = qOffsets[sequenceLength - 1];

    const int minROffset =
        min(min(rOffsets[0], rOffsets[sequenceLength - 1]),
            min(rOffsets[(nTrajectories - 1) * sequenceLength],
                rOffsets[nTrajectories * sequenceLength - 1]));
    const int maxROffset =
        max(max(rOffsets[0], rOffsets[sequenceLength - 1]),
            max(rOffsets[(nTrajectories - 1) * sequenceLength],
                rOffsets[nTrajectories * sequenceLength - 1]));

    for (unsigned int r = rBase; r < nRef; r += nThreads) {
        float val = 0.0f;
        if (q >= -minQOffset && q < nQue - maxQOffset && r >= -minROffset &&
            r < nRef - maxROffset) {
            float best = FLT_MAX;
            for (int i = 0; i < nTrajectories; i++) {
                float score = 0.0f;
                for (int j = 0; j < sequenceLength; j++) {
                    const int qOffset = qOffsets[j];
                    const int rOffset = rOffsets[i * sequenceLength + j];
                    score += diffMx[(q + qOffset) * nRef + r + rOffset];
                }
                best = fmin(best, score);
            }
            val = best;
        }
        sequenceStrengthOut[q * nRef + r] = val;
    }
}
