#pragma once

#include <Accelerate/Accelerate.h>
#include <complex>
#include <span>
#include <stdexcept>
#include <vector>

namespace zlbenchmark {
    template <typename F>
    class VDSPFFT;

    template <>
    class VDSPFFT<float> final {
        using C = std::complex<float>;

    public:
        explicit VDSPFFT(const size_t order) :
            order_(order), n_(1 << order), fft_setup_(vDSP_create_fftsetup(order, FFT_RADIX2)) {
            if (!fft_setup_) {
                throw std::runtime_error("vDSP_create_fftsetup failed");
            }
            temp_split_real_.resize(n_);
            temp_split_imag_.resize(n_);
            split_complex_.realp = temp_split_real_.data();
            split_complex_.imagp = temp_split_imag_.data();
        }

        ~VDSPFFT() {
            if (fft_setup_) {
                vDSP_destroy_fftsetup(fft_setup_);
            }
        }

        void forward(std::span<C> in_buffer, std::span<C> out_buffer) {
            vDSP_ctoz(reinterpret_cast<DSPComplex*>(in_buffer.data()), 2, &split_complex_, 1, n_);

            vDSP_fft_zip(fft_setup_, &split_complex_, 1, order_, FFT_FORWARD);

            vDSP_ztoc(&split_complex_, 1, reinterpret_cast<DSPComplex*>(out_buffer.data()), 2, n_);
        }

    private:
        size_t order_;
        size_t n_;
        FFTSetup fft_setup_;
        std::vector<float> temp_split_real_;
        std::vector<float> temp_split_imag_;
        DSPSplitComplex split_complex_;
    };

    template <>
    class VDSPFFT<double> final {
        using C = std::complex<double>;

    public:
        explicit VDSPFFT(const size_t order) :
            order_(order), n_(1 << order), fft_setup_(vDSP_create_fftsetupD(order, FFT_RADIX2)) {
            if (!fft_setup_) {
                throw std::runtime_error("vDSP_create_fftsetupD failed");
            }
            temp_split_real_.resize(n_);
            temp_split_imag_.resize(n_);
            split_complex_.realp = temp_split_real_.data();
            split_complex_.imagp = temp_split_imag_.data();
        }

        ~VDSPFFT() {
            if (fft_setup_) {
                vDSP_destroy_fftsetupD(fft_setup_);
            }
        }

        void forward(std::span<C> in_buffer, std::span<C> out_buffer) {
            vDSP_ctozD(reinterpret_cast<DSPDoubleComplex*>(in_buffer.data()), 2, &split_complex_, 1, n_);
            vDSP_fft_zipD(fft_setup_, &split_complex_, 1, order_, FFT_FORWARD);
            vDSP_ztocD(&split_complex_, 1, reinterpret_cast<DSPDoubleComplex*>(out_buffer.data()), 2, n_);
        }

    private:
        size_t order_;
        size_t n_;
        FFTSetupD fft_setup_;
        std::vector<double> temp_split_real_;
        std::vector<double> temp_split_imag_;
        DSPDoubleSplitComplex split_complex_;
    };
}
