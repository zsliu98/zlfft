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
    class SIMDStockhamRadix2Kernel24 {
        using C = std::complex<F>;

    public:
        explicit SIMDStockhamRadix2Kernel24(const size_t order) {
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

            kernel12(in, out, n);
            kernel4(out, in, n);
            w_ptr += 7;
            for (size_t half = 8; half < n; half <<= 1) {
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

        void kernel12(const C* __restrict in, C* __restrict out, const size_t n) {
            const size_t quarter_n = n >> 2;
            const size_t half_n = n >> 1;
            const size_t three_quarter_n = n - quarter_n;

            const F* __restrict group_in_0 = reinterpret_cast<const F*>(in);
            const F* __restrict group_in_1 = reinterpret_cast<const F*>(in) + (quarter_n << 1);
            const F* __restrict group_in_2 = reinterpret_cast<const F*>(in) + (half_n << 1);
            const F* __restrict group_in_3 = reinterpret_cast<const F*>(in) + (three_quarter_n << 1);
            F* __restrict out_ptr_base = reinterpret_cast<F*>(out);

            const hn::ScalableTag<F> d;
            const size_t lanes = hn::Lanes(d);

            size_t j = 0;

            // Ensure we process full vectors
            for (; j + lanes <= quarter_n; j += lanes) {
                using V = hn::Vec<decltype(d)>;
                V x0_r, x0_i, x1_r, x1_i, x2_r, x2_i, x3_r, x3_i;

                const auto j_shift = j << 1;

                hn::LoadInterleaved2(d, group_in_0 + j_shift, x0_r, x0_i);
                hn::LoadInterleaved2(d, group_in_1 + j_shift, x1_r, x1_i);
                hn::LoadInterleaved2(d, group_in_2 + j_shift, x2_r, x2_i);
                hn::LoadInterleaved2(d, group_in_3 + j_shift, x3_r, x3_i);

                const auto t0_r = hn::Add(x0_r, x2_r);
                const auto t0_i = hn::Add(x0_i, x2_i);
                const auto t1_r = hn::Sub(x0_r, x2_r);
                const auto t1_i = hn::Sub(x0_i, x2_i);

                const auto t2_r = hn::Add(x1_r, x3_r);
                const auto t2_i = hn::Add(x1_i, x3_i);
                const auto t3_r = hn::Sub(x1_r, x3_r);
                const auto t3_i = hn::Sub(x1_i, x3_i);

                const auto t3_neg_i_r = t3_i;
                const auto t3_neg_i_i = hn::Neg(t3_r);

                const auto out0_r = hn::Add(t0_r, t2_r);
                const auto out0_i = hn::Add(t0_i, t2_i);

                const auto out1_r = hn::Add(t1_r, t3_neg_i_r);
                const auto out1_i = hn::Add(t1_i, t3_neg_i_i);

                const auto out2_r = hn::Sub(t0_r, t2_r);
                const auto out2_i = hn::Sub(t0_i, t2_i);

                const auto out3_r = hn::Sub(t1_r, t3_neg_i_r);
                const auto out3_i = hn::Sub(t1_i, t3_neg_i_i);

                const auto r02_lo = hn::ZipLower(d, out0_r, out2_r);
                const auto r02_hi = hn::ZipUpper(d, out0_r, out2_r);
                const auto i02_lo = hn::ZipLower(d, out0_i, out2_i);
                const auto i02_hi = hn::ZipUpper(d, out0_i, out2_i);

                const auto r13_lo = hn::ZipLower(d, out1_r, out3_r);
                const auto r13_hi = hn::ZipUpper(d, out1_r, out3_r);
                const auto i13_lo = hn::ZipLower(d, out1_i, out3_i);
                const auto i13_hi = hn::ZipUpper(d, out1_i, out3_i);

                const size_t out_offset = j << 3;
                F* curr_out = out_ptr_base + out_offset;

                hn::StoreInterleaved4(r02_lo, i02_lo, r13_lo, i13_lo, d, curr_out);
                hn::StoreInterleaved4(r02_hi, i02_hi, r13_hi, i13_hi, d, curr_out + (lanes << 2));
            }

            for (; j < quarter_n; ++j) {
                const C x0 = in[j];
                const C x1 = in[quarter_n + j];
                const C x2 = in[half_n + j];
                const C x3 = in[three_quarter_n + j];

                const C t0 = x0 + x2;
                const C t1 = x0 - x2;
                const C t2 = x1 + x3;
                const C t3 = x1 - x3;
                const C t3_neg_i = {t3.imag(), -t3.real()};

                C* __restrict out_p = &out[j << 2];
                out_p[0] = t0 + t2;
                out_p[1] = t1 + t3_neg_i;
                out_p[2] = t0 - t2;
                out_p[3] = t1 - t3_neg_i;
            }
        }

        void kernel4(C* __restrict in, C* __restrict out, const size_t n) {
            static constexpr F kInvSqrt2 = static_cast<F>(1.0 / std::numbers::sqrt2);
            const C* __restrict in_a = &in[0];
            const C* __restrict in_b = &in[n >> 1];
            C* __restrict out_a = &out[0];
            C* __restrict out_b = &out[4];
            for (size_t j = 0; j < (n >> 1); j += 4) {
                const C* __restrict group_in_a = &in_a[j];
                const C* __restrict group_in_b = &in_b[j];
                C* __restrict group_out_a = &out_a[j << 1];
                C* __restrict group_out_b = &out_b[j << 1];
                { // twiddle = 1
                    const C a = group_in_a[0];
                    const C b = group_in_b[0];
                    group_out_a[0] = a + b;
                    group_out_b[0] = a - b;
                }
                { // twiddle = exp(-i * pi/4)
                    const C a = group_in_a[1];
                    const C b = group_in_b[1];
                    const C v = {
                        (b.real() + b.imag()) * kInvSqrt2,
                        (b.imag() - b.real()) * kInvSqrt2
                    };
                    group_out_a[1] = a + v;
                    group_out_b[1] = a - v;
                }
                { // twiddle = -i
                    const C a = group_in_a[2];
                    const C b = group_in_b[2];
                    const C v = {b.imag(), -b.real()};
                    group_out_a[2] = a + v;
                    group_out_b[2] = a - v;
                }
                { // twiddle = exp(-i * 3pi/4)
                    const C a = group_in_a[3];
                    const C b = group_in_b[3];
                    const C v = {
                        (b.imag() - b.real()) * kInvSqrt2,
                        (-b.real() - b.imag()) * kInvSqrt2
                    };
                    group_out_a[3] = a + v;
                    group_out_b[3] = a - v;
                }
            }
        }
    };
}
