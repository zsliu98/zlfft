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

#include "../source/naive_stockham_dit_radix2.hpp"


#ifdef USE_DOUBLE
using F = double;
#else
using F = float;
#endif
using C = std::complex<F>;

#ifdef ENABLE_KFR
#include "../kfr_impl/kfr_impl.hpp"
using FFTClass = zlbenchmark::KFRFFT<F>;
#else
using FFTClass = zlfft::NaiveStockhamDITRadix2<F>;
#endif

void generate_random_data(std::vector<C>& data) {
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<F> dist(static_cast<F>(-1.0), static_cast<F>(1.0));
    for (auto& x : data) {
        x = C(dist(gen), dist(gen));
    }
}