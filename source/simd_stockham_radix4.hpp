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
    class SIMDStockhamRadix4 {
        using C = std::complex<F>;

    public:
        explicit SIMDStockhamRadix4(const size_t order) :
            order_(order) {
            const auto n = static_cast<size_t>(1) << order;
            for (size_t width = (order_ % 2 == 0) ? 4 : 8; width < n; width <<= 2) {
                const double angle_step = -2.0 * std::numbers::pi / static_cast<double>(width << 2);
                for (size_t k = 0; k < width; ++k) {
                    const double angle = static_cast<double>(k) * angle_step;
                    twiddles_.emplace_back(static_cast<F>(std::cos(angle)), static_cast<F>(std::sin(angle)));
                }
                for (size_t k = 0; k < width; ++k) {
                    const double angle = static_cast<double>(k) * angle_step * 2.0;
                    twiddles_.emplace_back(static_cast<F>(std::cos(angle)), static_cast<F>(std::sin(angle)));
                }
                for (size_t k = 0; k < width; ++k) {
                    const double angle = static_cast<double>(k) * angle_step * 3.0;
                    twiddles_.emplace_back(static_cast<F>(std::cos(angle)), static_cast<F>(std::sin(angle)));
                }
            }
        }

        void forward(std::span<C> in_buffer, std::span<C> out_buffer) {
            assert(in_buffer.size() == out_buffer.size());
            assert(in_buffer.data() != out_buffer.data());

            const auto n = in_buffer.size();
            C* __restrict in = in_buffer.data();
            C* __restrict out = out_buffer.data();
            const C* __restrict w_ptr = twiddles_.data();

            kernel12(in, out, n);
            if (order_ % 2 == 1) {
                kernel4(out, in, n);
            } else {
                std::swap(in, out);
            }
            for (size_t width = (order_ % 2 == 0) ? 4 : 8; width < n; width <<= 2) {
                radix4(in, out, n, width, w_ptr);
                std::swap(in, out);
                w_ptr += 3 * width;
            }

            if (in == in_buffer.data()) {
                std::copy(in_buffer.begin(), in_buffer.end(), out_buffer.begin());
            }
        }

    private:
        size_t order_;
        std::vector<C> twiddles_;

        static void kernel12(const C* __restrict in, C* __restrict out, const size_t n) {
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

        void radix4(const C* __restrict in, C* __restrict out,
                    const size_t n, const size_t width,
                    const C* __restrict w_ptr) {
            const size_t quarter_n = n >> 2;
            const size_t half_n = n >> 1;
            const size_t three_quarter_n = quarter_n * 3; // avoids dependency on n

            const hn::ScalableTag<F> d;
            const size_t lanes = hn::Lanes(d);
            const bool use_simd = (width >= lanes);

            const C* __restrict const w1_base = w_ptr;
            const C* __restrict const w2_base = w_ptr + width;
            const C* __restrict const w3_base = w_ptr + 2 * width;

            const F* __restrict const w1_ptr = reinterpret_cast<const F*>(w1_base);
            const F* __restrict const w2_ptr = reinterpret_cast<const F*>(w2_base);
            const F* __restrict const w3_ptr = reinterpret_cast<const F*>(w3_base);

            for (size_t j = 0; j < quarter_n; j += width) {
                const C* __restrict in_0 = in + j;
                const C* __restrict in_1 = in + j + quarter_n;
                const C* __restrict in_2 = in + j + half_n;
                const C* __restrict in_3 = in + j + three_quarter_n;

                C* __restrict out_0 = out + (j << 2);
                C* __restrict out_1 = out_0 + width;
                C* __restrict out_2 = out_1 + width;
                C* __restrict out_3 = out_2 + width;

                size_t k = 0;

                if (use_simd) {
                    for (; k <= width - lanes; k += lanes) {
                        using V = hn::Vec<decltype(d)>;
                        V r0, i0, r1, i1, r2, i2, r3, i3;
                        V w1_r, w1_i, w2_r, w2_i, w3_r, w3_i;

                        hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in_0 + k), r0, i0);
                        hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in_1 + k), r1, i1);
                        hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in_2 + k), r2, i2);
                        hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in_3 + k), r3, i3);

                        hn::LoadInterleaved2(d, w1_ptr + (k << 1), w1_r, w1_i);
                        hn::LoadInterleaved2(d, w2_ptr + (k << 1), w2_r, w2_i);
                        hn::LoadInterleaved2(d, w3_ptr + (k << 1), w3_r, w3_i);

                        const auto t1_r = hn::NegMulAdd(i1, w1_i, hn::Mul(r1, w1_r));
                        const auto t1_i = hn::MulAdd(i1, w1_r, hn::Mul(r1, w1_i));
                        const auto t2_r = hn::NegMulAdd(i2, w2_i, hn::Mul(r2, w2_r));
                        const auto t2_i = hn::MulAdd(i2, w2_r, hn::Mul(r2, w2_i));
                        const auto t3_r = hn::NegMulAdd(i3, w3_i, hn::Mul(r3, w3_r));
                        const auto t3_i = hn::MulAdd(i3, w3_r, hn::Mul(r3, w3_i));

                        const auto s0_r = hn::Add(r0, t2_r);
                        const auto s0_i = hn::Add(i0, t2_i);
                        const auto s1_r = hn::Sub(r0, t2_r);
                        const auto s1_i = hn::Sub(i0, t2_i);
                        const auto s2_r = hn::Add(t1_r, t3_r);
                        const auto s2_i = hn::Add(t1_i, t3_i);
                        const auto s3_r = hn::Sub(t1_r, t3_r);
                        const auto s3_i = hn::Sub(t1_i, t3_i);

                        hn::StoreInterleaved2(hn::Add(s0_r, s2_r), hn::Add(s0_i, s2_i),
                                              d, reinterpret_cast<F*>(out_0 + k));
                        hn::StoreInterleaved2(hn::Add(s1_r, s3_i), hn::Sub(s1_i, s3_r),
                                              d, reinterpret_cast<F*>(out_1 + k));
                        hn::StoreInterleaved2(hn::Sub(s0_r, s2_r), hn::Sub(s0_i, s2_i),
                                              d, reinterpret_cast<F*>(out_2 + k));
                        hn::StoreInterleaved2(hn::Sub(s1_r, s3_i), hn::Add(s1_i, s3_r),
                                              d, reinterpret_cast<F*>(out_3 + k));
                    }
                }

                for (; k < width; ++k) {
                    const C x0 = in_0[k];
                    const C x1 = in_1[k] * w1_base[k];
                    const C x2 = in_2[k] * w2_base[k];
                    const C x3 = in_3[k] * w3_base[k];

                    const C s0 = x0 + x2;
                    const C s1 = x0 - x2;
                    const C s2 = x1 + x3;
                    const C s3 = x1 - x3;
                    const C neg_i_s3 = {s3.imag(), -s3.real()};

                    out_0[k] = s0 + s2;
                    out_1[k] = s1 + neg_i_s3;
                    out_2[k] = s0 - s2;
                    out_3[k] = s1 - neg_i_s3;
                }
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
                {
                    // twiddle = 1
                    const C a = group_in_a[0];
                    const C b = group_in_b[0];
                    group_out_a[0] = a + b;
                    group_out_b[0] = a - b;
                }
                {
                    // twiddle = exp(-i * pi/4)
                    const C a = group_in_a[1];
                    const C b = group_in_b[1];
                    const C v = {
                        (b.real() + b.imag()) * kInvSqrt2,
                        (b.imag() - b.real()) * kInvSqrt2
                    };
                    group_out_a[1] = a + v;
                    group_out_b[1] = a - v;
                }
                {
                    // twiddle = -i
                    const C a = group_in_a[2];
                    const C b = group_in_b[2];
                    const C v = {b.imag(), -b.real()};
                    group_out_a[2] = a + v;
                    group_out_b[2] = a - v;
                }
                {
                    // twiddle = exp(-i * 3pi/4)
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
