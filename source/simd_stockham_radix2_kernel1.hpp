#pragma once

#include <vector>
#include <span>
#include <complex>
#include <cmath>
#include <algorithm>
#include <numbers>
#include <cassert>

#include <hwy/highway.h>

namespace zlfft {
    namespace hn = hwy::HWY_NAMESPACE;
    template <typename F>
    class SIMDStockhamRadix2Kernel1 {
        using C = std::complex<F>;

    public:
        explicit SIMDStockhamRadix2Kernel1(const size_t order) {
            const auto n = static_cast<size_t>(1) << order;
            twiddles_.reserve(n - 1);
            for (size_t half = 1; half < n; half <<= 1) {
                const auto angle_step = -2.0 * std::numbers::pi / static_cast<double>(half << 1);
                for (size_t k = 0; k < half; ++k) {
                    const auto angle = static_cast<double>(k) * angle_step;
                    twiddles_.emplace_back(static_cast<F>(std::cos(angle)),
                                           static_cast<F>(std::sin(angle)));
                }
            }
        }

        void forward(std::span<C> in_buffer, std::span<C> out_buffer) {
            // in/out buffer size should match the init size
            assert(in_buffer.size() == twiddles_.size() + 1);
            // in/out buffer should have the same size
            assert(in_buffer.size() == out_buffer.size());
            // in/out buffer should be out-of-place
            assert(in_buffer.data() != out_buffer.data());
            const auto n = in_buffer.size();
            C* __restrict in = in_buffer.data();
            C* __restrict out = out_buffer.data();
            const C* __restrict w_ptr = twiddles_.data();

            const hn::ScalableTag<F> d;
            const size_t lanes = hn::Lanes(d);

            kernel1(in, out, n);
            w_ptr += 1;
            std::swap(in, out);
            for (size_t half = 2; half < n; half <<= 1) {
                for (size_t j = 0; j < (n >> 1); j += half) {
                    const C* __restrict group_in_a = &in[j];
                    const C* __restrict group_in_b = &in[j + (n >> 1)];
                    C* __restrict group_out_a = &out[j << 1];
                    C* __restrict group_out_b = &out[(j << 1) + half];

                    size_t k = 0;

                    if (half > lanes) {
                        for (; k <= half - lanes; k += lanes) {
                            const F* a_ptr = reinterpret_cast<const F*>(group_in_a + k);
                            const F* b_ptr = reinterpret_cast<const F*>(group_in_b + k);
                            const F* w_ptr_loc = reinterpret_cast<const F*>(w_ptr + k);

                            F* out_a_ptr = reinterpret_cast<F*>(group_out_a + k);
                            F* out_b_ptr = reinterpret_cast<F*>(group_out_b + k);

                            using V = hn::Vec<decltype(d)>;
                            V a_re, a_im, b_re, b_im, w_re, w_im;

                            hn::LoadInterleaved2(d, a_ptr, a_re, a_im);
                            hn::LoadInterleaved2(d, b_ptr, b_re, b_im);
                            hn::LoadInterleaved2(d, w_ptr_loc, w_re, w_im);

                            auto v_re = hn::NegMulAdd(w_im, b_im, hn::Mul(w_re, b_re));

                            auto v_im = hn::MulAdd(w_re, b_im, hn::Mul(w_im, b_re));

                            auto out_a_re = hn::Add(a_re, v_re);
                            auto out_a_im = hn::Add(a_im, v_im);

                            auto out_b_re = hn::Sub(a_re, v_re);
                            auto out_b_im = hn::Sub(a_im, v_im);

                            hn::StoreInterleaved2(out_a_re, out_a_im, d, out_a_ptr);
                            hn::StoreInterleaved2(out_b_re, out_b_im, d, out_b_ptr);
                        }
                    }
                    for (; k < half; ++k) {
                        const auto a = group_in_a[k];
                        const auto b = group_in_b[k];
                        const auto v = w_ptr[k] * b;

                        group_out_a[k] = a + v;
                        group_out_b[k] = a - v;
                    }
                }
                w_ptr += half;
                std::swap(in, out);
            }
            if (in == in_buffer.data()) {
                std::copy(in_buffer.begin(), in_buffer.end(), out_buffer.begin());
            }
        }

    private:
        std::vector<C> twiddles_;

        void kernel1(C* __restrict in, C* __restrict out, const size_t n) {
            using D = hn::ScalableTag<F>;
            const D d;
            const size_t lanes = hn::Lanes(d);

            const size_t half_n = n >> 1;

            const F* __restrict in_ptr_a = reinterpret_cast<const F*>(in);
            const F* __restrict in_ptr_b = reinterpret_cast<const F*>(in + half_n);
            F* __restrict out_ptr = reinterpret_cast<F*>(out);

            size_t j = 0;
            if (half_n >= lanes) {
                for (; j <= half_n - lanes; j += lanes) {
                    using V = hn::Vec<decltype(d)>;
                    V a_re, a_im, b_re, b_im;

                    hn::LoadInterleaved2(d, in_ptr_a + (j << 1), a_re, a_im);
                    hn::LoadInterleaved2(d, in_ptr_b + (j << 1), b_re, b_im);

                    auto sum_re = hn::Add(a_re, b_re);
                    auto sum_im = hn::Add(a_im, b_im);
                    auto diff_re = hn::Sub(a_re, b_re);
                    auto diff_im = hn::Sub(a_im, b_im);

                    hn::StoreInterleaved4(sum_re, sum_im, diff_re, diff_im, d, out_ptr + (j << 2));
                }
            }
            for (; j < half_n; ++j) {
                const auto a = in[j];
                const auto b = in[j + half_n];
                out[j << 1] = a + b;
                out[(j << 1) + 1] = a - b;
            }
        }
    };
}
