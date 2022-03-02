/**
 * @file
 */
#include <cmath>
#include <functional>
#include <mutex>
#include <vector>
#include <list>
#include <utility>
#include "ListAllocator.h"
#include "Helper.h"
#include <cuda_runtime.h>

#pragma once

namespace groupallocator {

    /**
     * Allocates memory in a page set by page_size.
     */
    class InPageAllocator {
    public:
        InPageAllocator() = delete;

        /**
         * Create the in page allocator.
         */
        InPageAllocator(std::size_t page_size)
                : PAGE_SIZE(page_size) {
            gpuAssert(cudaMallocManaged((void **) &mem, PAGE_SIZE), __FILE__, __LINE__);
            l = ListAllocator(mem, PAGE_SIZE);
        }

        /**
         * Deletes in page allocator, cuda context must exist to do so.
         */
        ~InPageAllocator() { gpuAssert(cudaFree(mem), __FILE__, __LINE__); }

        /**
         * Allocates memory of type T and sets *ptr to this memory of size s
         * @param ptr
         * @param s
         */
        template<class T>
        void allocate(T **ptr, size_t s, bool forceAligned128) {
#ifdef DEBUGALLOC
            std::clog << "Allocating in IPA " << __FILE__ << ":" << __LINE__ << std::endl;
#endif
            m.lock();
            l.alloc(ptr, s, forceAligned128);
            m.unlock();
        }

        template<class T>
        void free(T *ptr) {
            m.lock();
            l.free(ptr);
            m.unlock();
        }

        bool contains(size_t ptr) {
            return ptr >= (size_t) mem && ptr < (size_t) mem + PAGE_SIZE;
        }

        void moveToDevice(int device, cudaStream_t stream) {
            #ifndef DISABLE_PREFETCH
            gpuAssert(cudaMemPrefetchAsync(mem, PAGE_SIZE, device, stream), __FILE__, __LINE__);
            #endif
        }

        size_t getPages() { return 1; }

        size_t getPageSize() { return PAGE_SIZE; }

    private:
        char *mem;
        ListAllocator l;
        std::mutex m;
        const size_t PAGE_SIZE;
    };

}  // namespace groupallocator
