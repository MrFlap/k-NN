#ifndef KNNPLUGIN_JNI_HUGE_MEMORY_BUFFER_H
#define KNNPLUGIN_JNI_HUGE_MEMORY_BUFFER_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <stdexcept>

#ifdef __linux__
#include <sys/mman.h>
#endif

namespace knn_jni {

    // A grow-only buffer for large allocations. Not thread-safe.
    // - Uses anonymous mmap on Linux to bypass heap allocator retention
    // - Falls back to aligned new[] on other platforms
    class HugeMemoryBuffer {
    public:
        explicit HugeMemoryBuffer(size_t capacity)
            : capacity_(capacity), size_(0), data_(nullptr) {

            if (capacity_ == 0) {
                return;
            }

#ifdef __linux__
            data_ = static_cast<uint8_t*>(
                mmap(nullptr, capacity_, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
            );

            if (data_ == MAP_FAILED) {
                data_ = nullptr;
                throw std::runtime_error("mmap failed");
            }

#ifdef MADV_HUGEPAGE
            (void)madvise(data_, capacity_, MADV_HUGEPAGE);
#endif
#else
            data_ = static_cast<uint8_t*>(
                ::operator new[](capacity_, std::align_val_t(kAlignment))
            );
#endif
        }

        ~HugeMemoryBuffer() {
            release();
        }

        HugeMemoryBuffer(const HugeMemoryBuffer&) = delete;
        HugeMemoryBuffer& operator=(const HugeMemoryBuffer&) = delete;

        HugeMemoryBuffer(HugeMemoryBuffer&& other) noexcept
            : capacity_(other.capacity_),
              size_(other.size_),
              data_(other.data_) {
            other.data_ = nullptr;
            other.capacity_ = 0;
            other.size_ = 0;
        }

        HugeMemoryBuffer& operator=(HugeMemoryBuffer&& other) noexcept {
            if (this != &other) {
                release();

                capacity_ = other.capacity_;
                size_ = other.size_;
                data_ = other.data_;

                other.data_ = nullptr;
                other.capacity_ = 0;
                other.size_ = 0;
            }
            return *this;
        }

        uint8_t* data() { return data_; }
        const uint8_t* data() const { return data_; }

        size_t size() const { return size_; }
        size_t capacity() const { return capacity_; }

        void append(const uint8_t* src, size_t length) {
            if (length > capacity_ - size_) {
                throw std::overflow_error("append exceeds capacity");
            }

            std::memcpy(data_ + size_, src, length);
            size_ += length;
        }

    private:
        static constexpr size_t kAlignment = 4096;

        void release() {
            if (!data_) {
                return;
            }

#ifdef __linux__
            munmap(data_, capacity_);
#else
            ::operator delete[](data_, std::align_val_t(kAlignment));
#endif

            data_ = nullptr;
            capacity_ = 0;
            size_ = 0;
        }

        size_t capacity_;
        size_t size_;
        uint8_t* data_;
    };

}  // namespace knn_jni

#endif // KNNPLUGIN_JNI_HUGE_MEMORY_BUFFER_H
