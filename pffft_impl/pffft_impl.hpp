#pragma once

#include <span>
#include <vector>
#include <complex>
#include <cassert>

#include "pffft/include/pffft/pffft.h"
#include "pffft/include/pffft/pffft_double.h"

namespace zlbenchmark {
    template <typename F>
    class PffftFFT;

    template <>
    class PffftFFT<double> final {
        using C = std::complex<double>;

    public:
        explicit PffftFFT(const size_t order) :
            size_(1 << order) {
            setup_ = pffftd_new_setup(static_cast<int>(size_), PFFFT_COMPLEX);
            assert(setup_ != nullptr);

            work_ = static_cast<double*>(pffft_aligned_malloc(size_ * 2 * sizeof(double)));
        }

        ~PffftFFT() {
            if (work_)
                pffft_aligned_free(work_);
            if (setup_)
                pffftd_destroy_setup(setup_);
        }

        void forward(std::span<C> in_buffer, std::span<C> out_buffer) {
            auto* in_ptr = reinterpret_cast<const double*>(in_buffer.data());
            auto* out_ptr = reinterpret_cast<double*>(out_buffer.data());

            pffftd_transform_ordered(setup_, in_ptr, out_ptr, work_, PFFFT_FORWARD);
        }

    private:
        size_t size_;
        PFFFTD_Setup* setup_ = nullptr;
        double* work_ = nullptr;
    };

    template <>
    class PffftFFT<float> final {
        using C = std::complex<float>;

    public:
        explicit PffftFFT(const size_t order) :
            size_(1 << order) {
            setup_ = pffft_new_setup(static_cast<int>(size_), PFFFT_COMPLEX);
            assert(setup_ != nullptr);

            work_ = static_cast<float*>(pffft_aligned_malloc(size_ * 2 * sizeof(float)));
        }

        ~PffftFFT() {
            if (work_)
                pffft_aligned_free(work_);
            if (setup_)
                pffft_destroy_setup(setup_);
        }

        void forward(std::span<C> in_buffer, std::span<C> out_buffer) {
            auto* in_ptr = reinterpret_cast<const float*>(in_buffer.data());
            auto* out_ptr = reinterpret_cast<float*>(out_buffer.data());

            pffft_transform_ordered(setup_, in_ptr, out_ptr, work_, PFFFT_FORWARD);
        }

    private:
        size_t size_;
        PFFFT_Setup* setup_ = nullptr;
        float* work_ = nullptr;
    };

}
