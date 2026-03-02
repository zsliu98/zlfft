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
#elif defined(ENABLE_NAIVE_COOLEY_RADIX2)
#include "../source/naive_cooley_radix2.hpp"
using FFTClass = zlfft::NaiveCooleyRadix2<F>;
#elif defined(ENABLE_STOCKHAM_RADIX2_KERNEL4)
#include "../source/stockham_radix2_kernel4.hpp"
using FFTClass = zlfft::StockhamRadix2Kernel4<F>;
#else
using FFTClass = zlfft::NaiveStockhamRadix2<F>;
#endif

inline void generate_random_data(std::vector<C>& data) {
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<F> dist(static_cast<F>(-1.0), static_cast<F>(1.0));
    for (auto& x : data) {
        x = C(dist(gen), dist(gen));
    }
}