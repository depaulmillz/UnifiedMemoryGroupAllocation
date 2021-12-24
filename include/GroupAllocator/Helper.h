#include <cuda_runtime.h>
#include <cstdio>

#pragma once

namespace groupallocator {

    /**
     * Assert that CUDA returned successful
     * @param code
     * @param file
     * @param line
     * @param abort
     */
    inline void gpuAssert(cudaError_t code, const char *file, int line,
                          bool abort = true) {
        if (code != cudaSuccess) {
            fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                    line);
            if (abort)
                exit(code);
        }
    }

    }  // namespace groupallocator

