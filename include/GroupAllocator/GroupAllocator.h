#pragma once

#include <cmath>
#include <functional>
#include <mutex>
#include <vector>
#include <list>
#include <utility>
#include "MultiPageAllocator.h"
#include "InPageAllocator.h"

namespace groupallocator {

    /**
     * Allocates with group affinity
     */
    class GroupAllocator {
    public:

        GroupAllocator() = delete;

        /**
         * Constructor takes group_num to allocate to
         * @param group_num
         */
        GroupAllocator(int group_num, std::size_t page_size)
                : group_num_(group_num),
                  PAGE_SIZE(page_size) {
            mpa = new MultiPageAllocator(page_size);
        }

        /**
         * Delete function
         */
        ~GroupAllocator() {}

        /**
         * Function to free all memory of group allocator
         */
        void freeall() {
            for (auto &e : ipas) {
                delete e;
            }
            delete mpa;
            ipas = std::vector<InPageAllocator *>{};
            mpa = nullptr;
        }

        /**
         * Free pointer T* ptr
         * @tparam T
         * @param ptr
         */
        template<class T>
        void free(T *ptr) {
            mpa->free(ptr);
            m.lock();
            for (auto &e : ipas) {
                if (e->contains((size_t) ptr)) {
                    e->free(ptr);
                    break;
                }
            }
            m.unlock();
        }

        /**
         * Allocates memory of type T and sets *ptr to this memory of size s
         * @tparam T
         * @param ptr
         * @param s
         * @param forceAligned128
         */
        template<class T>
        void allocate(T **ptr, size_t s, bool forceAligned128) {
            if (ptr == NULL || s == 0) {
                return;
            }

            if (s + alignof(T *) > PAGE_SIZE) {
                mpa->allocate<T>(ptr, s, forceAligned128);
            } else {
                m.lock();
                int lastSize = ipas.size();
                if (lastSize == 0) {
                    InPageAllocator *ipa_new =
                            new InPageAllocator(PAGE_SIZE);
                    ipas.push_back(ipa_new);
                }
                auto ipa3 = ipas[ipas.size() - 1];
                m.unlock();
                ipa3->allocate<T>(ptr, s, forceAligned128);
                while (*ptr == NULL) {
                    InPageAllocator *ipa2 =
                            new InPageAllocator(PAGE_SIZE);
                    m.lock();
                    if (lastSize == ipas.size()) {
                        ipas.push_back(ipa2);
                        lastSize = ipas.size();
                    }
                    m.unlock();
                    m.lock();
                    auto ipa = ipas[ipas.size() - 1];
                    m.unlock();
                    ipa->allocate<T>(ptr, s, forceAligned128);
                }
            }
        }

        /**
         * Move to memory device in stream
         * @param device
         * @param stream
         */
        void moveToDevice(int device, cudaStream_t stream) {
            mpa->moveToDevice(device, stream);
            m.lock();
            for (auto &e : ipas) {
                e->moveToDevice(device, stream);
            }
            m.unlock();
        }

        size_t pagesAllocated() {
            auto s = mpa->getPages();
            m.lock();
            s += ipas.size();
            m.unlock();
            return s;
        }

        size_t getPageSize() {
            return PAGE_SIZE;
        }

    private:
        std::vector<InPageAllocator *> ipas;
        MultiPageAllocator *mpa;
        std::mutex m;
        int group_num_;
        const size_t PAGE_SIZE;
    };

}  // namespace groupallocator
