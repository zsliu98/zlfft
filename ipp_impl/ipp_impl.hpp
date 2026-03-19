#pragma once

#include <complex>
#include <mutex>
#include <span>
#include <stdexcept>
#include <vector>

#include <ippcore.h>
#include <ipps.h>

namespace zlbenchmark {
    enum class IPPSimdLevel {
        Auto,
        SSE42,
        AVX2,
        AVX512
    };

    inline void init_ipp_dispatcher(IPPSimdLevel target_level = IPPSimdLevel::Auto) {
        static std::once_flag flag;
        std::call_once(flag, [target_level]() {
            if (target_level == IPPSimdLevel::Auto) {
                ippInit();
                return;
            }

            Ipp64u cpuFeatures;

            ippGetCpuFeatures(&cpuFeatures, 0);

            if (target_level <= IPPSimdLevel::AVX2) {
                cpuFeatures &= ~(ippCPUID_AVX512F | ippCPUID_AVX512CD |
                                 ippCPUID_AVX512VL | ippCPUID_AVX512BW | ippCPUID_AVX512DQ);
            }

            if (target_level <= IPPSimdLevel::SSE42) {
                cpuFeatures &= ~(ippCPUID_AVX | ippCPUID_AVX2);
            }

            IppStatus status = ippSetCpuFeatures(cpuFeatures);
            if (status != ippStsNoErr) {
                throw std::runtime_error("Failed to explicitly set IPP CPU features.");
            }
        });
    }

    template <typename F>
    class IPPFFT;

    template <>
    class IPPFFT<float> final {
        using C = std::complex<float>;

    public:
        explicit IPPFFT(const int order) : order_(order), n_(1 << order) {
            init_ipp_dispatcher(IPPSimdLevel::AVX2);
            int specSize = 0, initSize = 0, bufSize = 0;

            IppStatus status = ippsFFTGetSize_C_32fc(order_, IPP_FFT_DIV_INV_BY_N, ippAlgHintNone, &specSize, &initSize, &bufSize);
            if (status != ippStsNoErr) {
                throw std::runtime_error("ippsFFTGetSize_C_32fc failed");
            }

            spec_mem_ = ippsMalloc_8u(specSize);
            work_buf_ = bufSize > 0 ? ippsMalloc_8u(bufSize) : nullptr;
            Ipp8u* init_buf = initSize > 0 ? ippsMalloc_8u(initSize) : nullptr;

            status = ippsFFTInit_C_32fc(&spec_, order_, IPP_FFT_DIV_INV_BY_N, ippAlgHintNone, spec_mem_, init_buf);

            if (init_buf) {
                ippsFree(init_buf);
            }

            if (status != ippStsNoErr) {
                if (spec_mem_) ippsFree(spec_mem_);
                if (work_buf_) ippsFree(work_buf_);
                throw std::runtime_error("ippsFFTInit_C_32fc failed");
            }
        }

        ~IPPFFT() {
            if (spec_mem_) {
                ippsFree(spec_mem_);
            }
            if (work_buf_) {
                ippsFree(work_buf_);
            }
        }

        void forward(std::span<const C> in_buffer, std::span<C> out_buffer) {
            ippsFFTFwd_CToC_32fc(
                reinterpret_cast<const Ipp32fc*>(in_buffer.data()),
                reinterpret_cast<Ipp32fc*>(out_buffer.data()),
                spec_,
                work_buf_
            );
        }

    private:
        int order_;
        int n_;
        IppsFFTSpec_C_32fc* spec_{nullptr};
        Ipp8u* spec_mem_{nullptr};
        Ipp8u* work_buf_{nullptr};
    };

    template <>
    class IPPFFT<double> final {
        using C = std::complex<double>;

    public:
        explicit IPPFFT(const int order) : order_(order), n_(1 << order) {
            init_ipp_dispatcher(IPPSimdLevel::AVX2);
            int specSize = 0, initSize = 0, bufSize = 0;
            
            IppStatus status = ippsFFTGetSize_C_64fc(order_, IPP_FFT_DIV_INV_BY_N, ippAlgHintNone, &specSize, &initSize, &bufSize);
            if (status != ippStsNoErr) {
                throw std::runtime_error("ippsFFTGetSize_C_64fc failed");
            }

            spec_mem_ = ippsMalloc_8u(specSize);
            work_buf_ = bufSize > 0 ? ippsMalloc_8u(bufSize) : nullptr;
            Ipp8u* init_buf = initSize > 0 ? ippsMalloc_8u(initSize) : nullptr;

            status = ippsFFTInit_C_64fc(&spec_, order_, IPP_FFT_DIV_INV_BY_N, ippAlgHintNone, spec_mem_, init_buf);
            
            if (init_buf) {
                ippsFree(init_buf);
            }

            if (status != ippStsNoErr) {
                if (spec_mem_) ippsFree(spec_mem_);
                if (work_buf_) ippsFree(work_buf_);
                throw std::runtime_error("ippsFFTInit_C_64fc failed");
            }
        }

        ~IPPFFT() {
            if (spec_mem_) {
                ippsFree(spec_mem_);
            }
            if (work_buf_) {
                ippsFree(work_buf_);
            }
        }

        void forward(std::span<const C> in_buffer, std::span<C> out_buffer) {
            ippsFFTFwd_CToC_64fc(
                reinterpret_cast<const Ipp64fc*>(in_buffer.data()),
                reinterpret_cast<Ipp64fc*>(out_buffer.data()),
                spec_,
                work_buf_
            );
        }

    private:
        int order_;
        int n_;
        IppsFFTSpec_C_64fc* spec_{nullptr};
        Ipp8u* spec_mem_{nullptr};
        Ipp8u* work_buf_{nullptr};
    };
}
