#pragma once

#include <chrono>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <random>
#include <span>
#include <string>
#include <vector>
#include <new>

#include "simd_stockham_radix2_kernel1.hpp"
#include "../source/naive_stockham_radix2.hpp"


#ifdef USE_DOUBLE
using F = double;
#else
using F = float;
#endif
using C = std::complex<F>;

#if defined(ENABLE_KFR)
#include "../kfr_impl/kfr_impl.hpp"
using FFTClass = zlbenchmark::KFRFFT<F>;
#elif defined(ENABLE_FFTW3)
#include "../fftw3_impl/fftw3_impl.hpp"
using FFTClass = zlbenchmark::FFTW3FFT<F>;
#elif defined(ENABLE_NAIVE_COOLEY_RADIX2)
#include "../source/naive_cooley_radix2.hpp"
using FFTClass = zlfft::NaiveCooleyRadix2<F>;
#elif defined(ENABLE_STOCKHAM_RADIX2_KERNEL24)
#include "../source/stockham_radix2_kernel24.hpp"
using FFTClass = zlfft::StockhamRadix2Kernel24<F>;
#elif defined(ENABLE_NAIVE_STOCKHAM_RADIX4)
#include "../source/naive_stockham_radix4.hpp"
using FFTClass = zlfft::NaiveStockhamRadix4<F>;
#elif defined(ENABLE_STOCKHAM_RADIX2_KERNEL248)
#include "../source/stockham_radix2_kernel248.hpp"
using FFTClass = zlfft::StockhamRadix2Kernel248<F>;
#elif defined(ENABLE_SIMD_STOCKHAM_RADIX2)
#include "../source/simd_stockham_radix2.hpp"
using FFTClass = zlfft::SIMDStockhamRadix2<F>;
#elif defined(ENABLE_SIMD_STOCKHAM_RADIX2_KERNEL1)
#include "../source/simd_stockham_radix2_kernel1.hpp"
using FFTClass = zlfft::SIMDStockhamRadix2Kernel1<F>;
#elif defined(ENABLE_SIMD_STOCKHAM_RADIX2_KERNEL2)
#include "../source/simd_stockham_radix2_kernel2.hpp"
using FFTClass = zlfft::SIMDStockhamRadix2Kernel2<F>;
#elif defined(ENABLE_SIMD_STOCKHAM_RADIX2_KERNEL24)
#include "../source/simd_stockham_radix2_kernel24.hpp"
using FFTClass = zlfft::SIMDStockhamRadix2Kernel24<F>;
#elif defined(ENABLE_SIMD_STOCKHAM_RADIX2_KERNEL248)
#include "../source/simd_stockham_radix2_kernel248.hpp"
using FFTClass = zlfft::SIMDStockhamRadix2Kernel248<F>;
#elif defined(ENABLE_SIMD_STOCKHAM_RADIX4)
#include "../source/simd_stockham_radix4.hpp"
using FFTClass = zlfft::SIMDStockhamRadix4<F>;
#else
using FFTClass = zlfft::NaiveStockhamRadix2<F>;
#endif

inline void generate_random_data(std::span<C> data) {
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<F> dist(static_cast<F>(-1.0), static_cast<F>(1.0));
    for (auto& x : data) {
        x = C(dist(gen), dist(gen));
    }
}

template <typename T, std::size_t Align = 64>
struct AlignedAllocator {
    using value_type = T;
    template <class U>
    struct rebind {
        using other = AlignedAllocator<U, Align>;
    };

    AlignedAllocator() = default;

    template <class U>
    constexpr AlignedAllocator(const AlignedAllocator<U, Align>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_alloc();

        void* ptr = ::operator new(n * sizeof(T), std::align_val_t(Align));
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept {
        ::operator delete(p, std::align_val_t(Align));
    }

    bool operator==(const AlignedAllocator&) const { return true; }
    bool operator!=(const AlignedAllocator&) const { return false; }
};
