#include <cmath>
#include <functional>
#include <mutex>
#include <vector>
#include <list>
#include <utility>
#include <cuda_runtime.h>
#include "Helper.h"

#pragma once

namespace groupallocator {

     /**
     * Allocates multiple pages of memory
     */
    class MultiPageAllocator {
    public:
        MultiPageAllocator() = delete;

        /**
         * Constructor
         */
        MultiPageAllocator(std::size_t page_size)
                : PAGE_SIZE(page_size), pagesAllocated(0) {}

        /**
         * Delete function
         */
        ~MultiPageAllocator() {
            m.lock();
            for (auto &e : mem) {
                gpuAssert(cudaFree((void *) e.first), __FILE__, __LINE__);
            }
            m.unlock();
        }

        /**
         * Allocates memory of type T and sets *ptr to this memory of size s
         * @param ptr
         * @param s
         */
        template<class T>
        void allocate(T **ptr, size_t s, bool forceAligned128) {
#ifdef DEBUGALLOC
            std::clog << "Allocating in MPA " << __FILE__ << ":" << __LINE__ << std::endl;
#endif
            size_t pages_needed = (size_t) ceil(s / (double) PAGE_SIZE);
            char *c;
            gpuAssert(cudaMallocManaged((void **) &c, pages_needed * PAGE_SIZE), __FILE__, __LINE__);
            *ptr = (T *) c;
            m.lock();
            pagesAllocated += pages_needed;
            mem.push_back({c, pages_needed * PAGE_SIZE});
            m.unlock();
        }

        template<class T>
        void free(T *ptr) {
            m.lock();
            for (auto i = mem.begin(); i != mem.end(); i++) {
                if ((size_t) i->first == (size_t) ptr) {
                    gpuAssert(cudaFree((void *) i->first), __FILE__, __LINE__);
                    mem.erase(i);
                    break;
                }
            }
            m.unlock();
        }

        void moveToDevice(int device, cudaStream_t stream) {
            #ifndef DISABLE_PREFETCH
            m.lock();
            for (auto i = mem.begin(); i != mem.end(); i++) {
                gpuAssert(cudaMemPrefetchAsync(i->first, i->second, device, stream), __FILE__, __LINE__);
            }
            m.unlock();
            #endif
        }

        size_t getPages() {
            std::unique_lock <std::mutex> ul(m);
            return pagesAllocated;
        }

        size_t getPageSize() { return PAGE_SIZE; }

    private:
        std::list <std::pair<char *, size_t>> mem;
        std::mutex m;
        const size_t PAGE_SIZE;
        size_t pagesAllocated;
    };

}  // namespace groupallocator
