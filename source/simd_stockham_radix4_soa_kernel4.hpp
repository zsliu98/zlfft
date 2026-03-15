#pragma once

#include <vector>
#include <span>
#include <complex>
#include <cmath>
#include <algorithm>
#include <numbers>
#include <cassert>

#include <hwy/highway.h>
#include <hwy/aligned_allocator.h>

namespace zlfft {
    namespace hn = hwy::HWY_NAMESPACE;

    template <typename F>
    class SIMDStockhamRadix4SOAKernel4 {
        using C = std::complex<F>;

    public:
        explicit SIMDStockhamRadix4SOAKernel4(const size_t order) :
            order_(order) {
            const auto n = static_cast<size_t>(1) << order;

            size_t num_twiddles = 0;
            for (size_t width = (order_ % 2 == 0) ? 4 : 8; width < n; width <<= 2) {
                num_twiddles += 3 * width;
            }

            twiddles_r_ = hwy::AllocateAligned<F>(num_twiddles);
            twiddles_i_ = hwy::AllocateAligned<F>(num_twiddles);

            size_t offset = 0;
            for (size_t width = (order_ % 2 == 0) ? 4 : 8; width < n; width <<= 2) {
                const double angle_step = -2.0 * std::numbers::pi / static_cast<double>(width << 2);
                for (size_t k = 0; k < width; ++k, ++offset) {
                    const double angle = static_cast<double>(k) * angle_step;
                    twiddles_r_[offset] = static_cast<F>(std::cos(angle));
                    twiddles_i_[offset] = static_cast<F>(std::sin(angle));
                }
                for (size_t k = 0; k < width; ++k, ++offset) {
                    const double angle = static_cast<double>(k) * angle_step * 2.0;
                    twiddles_r_[offset] = static_cast<F>(std::cos(angle));
                    twiddles_i_[offset] = static_cast<F>(std::sin(angle));
                }
                for (size_t k = 0; k < width; ++k, ++offset) {
                    const double angle = static_cast<double>(k) * angle_step * 3.0;
                    twiddles_r_[offset] = static_cast<F>(std::cos(angle));
                    twiddles_i_[offset] = static_cast<F>(std::sin(angle));
                }
            }

            const auto pad = (64 / sizeof(F)) + 16;
            stride_ = n + pad;
            workspace_ = hwy::AllocateAligned<F>(4 * stride_);
        }

        void forward(std::span<C> in_buffer, std::span<C> out_buffer) {
            assert(in_buffer.size() == out_buffer.size());
            assert(in_buffer.data() != out_buffer.data());

            const auto n = in_buffer.size();

            F* __restrict in_r = workspace_.get();
            F* __restrict in_i = in_r + stride_;
            F* __restrict out_r = in_i + stride_;
            F* __restrict out_i = out_r + stride_;

            const F* __restrict w_r_ptr = twiddles_r_.get();
            const F* __restrict w_i_ptr = twiddles_i_.get();

            if (order_ % 2 == 1) {
                kernel124_fused(in_buffer.data(), out_r, out_i, n);
            } else {
                kernel12_fused(in_buffer.data(), out_r, out_i, n);
            }
            std::swap(in_r, out_r);
            std::swap(in_i, out_i);

            size_t width = (order_ % 2 == 0) ? 4 : 8;

            while (width < (n >> 2)) {
                radix4(in_r, in_i, out_r, out_i, n, width, w_r_ptr, w_i_ptr);
                std::swap(in_r, out_r);
                std::swap(in_i, out_i);
                w_r_ptr += 3 * width;
                w_i_ptr += 3 * width;
                width <<= 2;
            }

            if (width < n) {
                radix4_fused_aos(in_r, in_i, out_buffer.data(), n, width, w_r_ptr, w_i_ptr);
            } else {
                soa_to_aos(in_r, in_i, out_buffer.data(), n);
            }
        }

    private:
        size_t order_;
        size_t stride_;
        hwy::AlignedFreeUniquePtr<F[]> twiddles_r_;
        hwy::AlignedFreeUniquePtr<F[]> twiddles_i_;
        hwy::AlignedFreeUniquePtr<F[]> workspace_;

        static void aos_to_soa(const C* __restrict in, F* __restrict out_r, F* __restrict out_i, size_t n) {
            const hn::ScalableTag<F> d;
            const size_t lanes = hn::Lanes(d);
            size_t i = 0;
            for (; i + lanes <= n; i += lanes) {
                hn::Vec<decltype(d)> r, im;
                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + i), r, im);
                hn::Store(r, d, out_r + i);
                hn::Store(im, d, out_i + i);
            }
            for (; i < n; ++i) {
                out_r[i] = in[i].real();
                out_i[i] = in[i].imag();
            }
        }

        static void soa_to_aos(const F* __restrict in_r, const F* __restrict in_i, C* __restrict out, size_t n) {
            const hn::ScalableTag<F> d;
            const size_t lanes = hn::Lanes(d);
            size_t i = 0;
            for (; i + lanes <= n; i += lanes) {
                auto r = hn::Load(d, in_r + i);
                auto im = hn::Load(d, in_i + i);
                hn::StoreInterleaved2(r, im, d, reinterpret_cast<F*>(out + i));
            }
            for (; i < n; ++i) {
                out[i] = C(in_r[i], in_i[i]);
            }
        }

        static void kernel12_fused(const C* __restrict in,
                                   F* __restrict out_r, F* __restrict out_i, const size_t n) {
            const size_t quarter_n = n >> 2;
            const size_t half_n = n >> 1;
            const size_t three_quarter_n = quarter_n * 3;

            const hn::ScalableTag<F> d;
            const size_t lanes = hn::Lanes(d);
            size_t j = 0;

            for (; j + lanes <= quarter_n; j += lanes) {
                hn::Vec<decltype(d)> x0_r, x0_i, x1_r, x1_i, x2_r, x2_i, x3_r, x3_i;

                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + j), x0_r, x0_i);
                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + j + quarter_n), x1_r, x1_i);
                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + j + half_n), x2_r, x2_i);
                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + j + three_quarter_n), x3_r, x3_i);

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

                const size_t out_offset = j << 2;
                hn::StoreInterleaved4(out0_r, out1_r, out2_r, out3_r, d, out_r + out_offset);
                hn::StoreInterleaved4(out0_i, out1_i, out2_i, out3_i, d, out_i + out_offset);
            }

            for (; j < quarter_n; ++j) {
                const F x0_r = in[j].real(), x0_i = in[j].imag();
                const F x1_r = in[j + quarter_n].real(), x1_i = in[j + quarter_n].imag();
                const F x2_r = in[j + half_n].real(), x2_i = in[j + half_n].imag();
                const F x3_r = in[j + three_quarter_n].real(), x3_i = in[j + three_quarter_n].imag();

                const F t0_r = x0_r + x2_r, t0_i = x0_i + x2_i;
                const F t1_r = x0_r - x2_r, t1_i = x0_i - x2_i;
                const F t2_r = x1_r + x3_r, t2_i = x1_i + x3_i;
                const F t3_r = x1_r - x3_r, t3_i = x1_i - x3_i;

                const F t3_neg_i_r = t3_i, t3_neg_i_i = -t3_r;

                const size_t out_idx = j << 2;
                out_r[out_idx + 0] = t0_r + t2_r;
                out_i[out_idx + 0] = t0_i + t2_i;
                out_r[out_idx + 1] = t1_r + t3_neg_i_r;
                out_i[out_idx + 1] = t1_i + t3_neg_i_i;
                out_r[out_idx + 2] = t0_r - t2_r;
                out_i[out_idx + 2] = t0_i - t2_i;
                out_r[out_idx + 3] = t1_r - t3_neg_i_r;
                out_i[out_idx + 3] = t1_i - t3_neg_i_i;
            }
        }

        void radix4(const F* __restrict in_r, const F* __restrict in_i,
                    F* __restrict out_r, F* __restrict out_i,
                    const size_t n, const size_t width,
                    const F* __restrict w_r_ptr, const F* __restrict w_i_ptr) {
            const size_t quarter_n = n >> 2;
            const size_t half_n = n >> 1;
            const size_t three_quarter_n = quarter_n * 3;

            const hn::ScalableTag<F> d;
            const size_t lanes = hn::Lanes(d);
            const bool use_simd = (width >= lanes);

            const F* __restrict const w1_r_base = w_r_ptr;
            const F* __restrict const w1_i_base = w_i_ptr;
            const F* __restrict const w2_r_base = w_r_ptr + width;
            const F* __restrict const w2_i_base = w_i_ptr + width;
            const F* __restrict const w3_r_base = w_r_ptr + 2 * width;
            const F* __restrict const w3_i_base = w_i_ptr + 2 * width;

            for (size_t j = 0; j < quarter_n; j += width) {
                const F* __restrict in_0_r = in_r + j;
                const F* __restrict in_0_i = in_i + j;
                const F* __restrict in_1_r = in_r + j + quarter_n;
                const F* __restrict in_1_i = in_i + j + quarter_n;
                const F* __restrict in_2_r = in_r + j + half_n;
                const F* __restrict in_2_i = in_i + j + half_n;
                const F* __restrict in_3_r = in_r + j + three_quarter_n;
                const F* __restrict in_3_i = in_i + j + three_quarter_n;

                F* __restrict out_0_r = out_r + (j << 2);
                F* __restrict out_0_i = out_i + (j << 2);
                F* __restrict out_1_r = out_0_r + width;
                F* __restrict out_1_i = out_0_i + width;
                F* __restrict out_2_r = out_1_r + width;
                F* __restrict out_2_i = out_1_i + width;
                F* __restrict out_3_r = out_2_r + width;
                F* __restrict out_3_i = out_2_i + width;

                size_t k = 0;

                if (use_simd) {
                    for (; k <= width - lanes; k += lanes) {
                        auto r0 = hn::Load(d, in_0_r + k);
                        auto i0 = hn::Load(d, in_0_i + k);
                        auto r1 = hn::Load(d, in_1_r + k);
                        auto i1 = hn::Load(d, in_1_i + k);
                        auto r2 = hn::Load(d, in_2_r + k);
                        auto i2 = hn::Load(d, in_2_i + k);
                        auto r3 = hn::Load(d, in_3_r + k);
                        auto i3 = hn::Load(d, in_3_i + k);

                        auto w1_r = hn::Load(d, w1_r_base + k);
                        auto w1_i = hn::Load(d, w1_i_base + k);
                        auto w2_r = hn::Load(d, w2_r_base + k);
                        auto w2_i = hn::Load(d, w2_i_base + k);
                        auto w3_r = hn::Load(d, w3_r_base + k);
                        auto w3_i = hn::Load(d, w3_i_base + k);

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

                        hn::Store(hn::Add(s0_r, s2_r), d, out_0_r + k);
                        hn::Store(hn::Add(s0_i, s2_i), d, out_0_i + k);
                        hn::Store(hn::Add(s1_r, s3_i), d, out_1_r + k);
                        hn::Store(hn::Sub(s1_i, s3_r), d, out_1_i + k);
                        hn::Store(hn::Sub(s0_r, s2_r), d, out_2_r + k);
                        hn::Store(hn::Sub(s0_i, s2_i), d, out_2_i + k);
                        hn::Store(hn::Sub(s1_r, s3_i), d, out_3_r + k);
                        hn::Store(hn::Add(s1_i, s3_r), d, out_3_i + k);
                    }
                }

                for (; k < width; ++k) {
                    const F x0_r = in_0_r[k], x0_i = in_0_i[k];
                    const F x1_r = in_1_r[k] * w1_r_base[k] - in_1_i[k] * w1_i_base[k];
                    const F x1_i = in_1_r[k] * w1_i_base[k] + in_1_i[k] * w1_r_base[k];
                    const F x2_r = in_2_r[k] * w2_r_base[k] - in_2_i[k] * w2_i_base[k];
                    const F x2_i = in_2_r[k] * w2_i_base[k] + in_2_i[k] * w2_r_base[k];
                    const F x3_r = in_3_r[k] * w3_r_base[k] - in_3_i[k] * w3_i_base[k];
                    const F x3_i = in_3_r[k] * w3_i_base[k] + in_3_i[k] * w3_r_base[k];

                    const F s0_r = x0_r + x2_r, s0_i = x0_i + x2_i;
                    const F s1_r = x0_r - x2_r, s1_i = x0_i - x2_i;
                    const F s2_r = x1_r + x3_r, s2_i = x1_i + x3_i;
                    const F s3_r = x1_r - x3_r, s3_i = x1_i - x3_i;

                    const F neg_i_s3_r = s3_i, neg_i_s3_i = -s3_r;

                    out_0_r[k] = s0_r + s2_r;
                    out_0_i[k] = s0_i + s2_i;
                    out_1_r[k] = s1_r + neg_i_s3_r;
                    out_1_i[k] = s1_i + neg_i_s3_i;
                    out_2_r[k] = s0_r - s2_r;
                    out_2_i[k] = s0_i - s2_i;
                    out_3_r[k] = s1_r - neg_i_s3_r;
                    out_3_i[k] = s1_i - neg_i_s3_i;
                }
            }
        }

        static void radix4_fused_aos(const F* __restrict in_r, const F* __restrict in_i,
                                     C* __restrict out,
                                     const size_t n, const size_t width,
                                     const F* __restrict w_r_ptr, const F* __restrict w_i_ptr) {
            const size_t quarter_n = n >> 2;
            const size_t half_n = n >> 1;
            const size_t three_quarter_n = quarter_n * 3;

            const hn::ScalableTag<F> d;
            const size_t lanes = hn::Lanes(d);
            const bool use_simd = (width >= lanes);

            const F* __restrict const w1_r_base = w_r_ptr;
            const F* __restrict const w1_i_base = w_i_ptr;
            const F* __restrict const w2_r_base = w_r_ptr + width;
            const F* __restrict const w2_i_base = w_i_ptr + width;
            const F* __restrict const w3_r_base = w_r_ptr + 2 * width;
            const F* __restrict const w3_i_base = w_i_ptr + 2 * width;

            for (size_t j = 0; j < quarter_n; j += width) {
                const F* __restrict in_0_r = in_r + j;
                const F* __restrict in_0_i = in_i + j;
                const F* __restrict in_1_r = in_r + j + quarter_n;
                const F* __restrict in_1_i = in_i + j + quarter_n;
                const F* __restrict in_2_r = in_r + j + half_n;
                const F* __restrict in_2_i = in_i + j + half_n;
                const F* __restrict in_3_r = in_r + j + three_quarter_n;
                const F* __restrict in_3_i = in_i + j + three_quarter_n;

                C* __restrict out_0 = out + (j << 2);
                C* __restrict out_1 = out_0 + width;
                C* __restrict out_2 = out_1 + width;
                C* __restrict out_3 = out_2 + width;

                size_t k = 0;

                if (use_simd) {
                    for (; k <= width - lanes; k += lanes) {
                        auto r0 = hn::Load(d, in_0_r + k);
                        auto i0 = hn::Load(d, in_0_i + k);
                        auto r1 = hn::Load(d, in_1_r + k);
                        auto i1 = hn::Load(d, in_1_i + k);
                        auto r2 = hn::Load(d, in_2_r + k);
                        auto i2 = hn::Load(d, in_2_i + k);
                        auto r3 = hn::Load(d, in_3_r + k);
                        auto i3 = hn::Load(d, in_3_i + k);

                        auto w1_r = hn::Load(d, w1_r_base + k);
                        auto w1_i = hn::Load(d, w1_i_base + k);
                        auto w2_r = hn::Load(d, w2_r_base + k);
                        auto w2_i = hn::Load(d, w2_i_base + k);
                        auto w3_r = hn::Load(d, w3_r_base + k);
                        auto w3_i = hn::Load(d, w3_i_base + k);

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

                        hn::StoreInterleaved2(hn::Add(s0_r, s2_r), hn::Add(s0_i, s2_i), d,
                                              reinterpret_cast<F*>(out_0 + k));
                        hn::StoreInterleaved2(hn::Add(s1_r, s3_i), hn::Sub(s1_i, s3_r), d,
                                              reinterpret_cast<F*>(out_1 + k));
                        hn::StoreInterleaved2(hn::Sub(s0_r, s2_r), hn::Sub(s0_i, s2_i), d,
                                              reinterpret_cast<F*>(out_2 + k));
                        hn::StoreInterleaved2(hn::Sub(s1_r, s3_i), hn::Add(s1_i, s3_r), d,
                                              reinterpret_cast<F*>(out_3 + k));
                    }
                }

                for (; k < width; ++k) {
                    const F x0_r = in_0_r[k], x0_i = in_0_i[k];
                    const F x1_r = in_1_r[k] * w1_r_base[k] - in_1_i[k] * w1_i_base[k];
                    const F x1_i = in_1_r[k] * w1_i_base[k] + in_1_i[k] * w1_r_base[k];
                    const F x2_r = in_2_r[k] * w2_r_base[k] - in_2_i[k] * w2_i_base[k];
                    const F x2_i = in_2_r[k] * w2_i_base[k] + in_2_i[k] * w2_r_base[k];
                    const F x3_r = in_3_r[k] * w3_r_base[k] - in_3_i[k] * w3_i_base[k];
                    const F x3_i = in_3_r[k] * w3_i_base[k] + in_3_i[k] * w3_r_base[k];

                    const F s0_r = x0_r + x2_r, s0_i = x0_i + x2_i;
                    const F s1_r = x0_r - x2_r, s1_i = x0_i - x2_i;
                    const F s2_r = x1_r + x3_r, s2_i = x1_i + x3_i;
                    const F s3_r = x1_r - x3_r, s3_i = x1_i - x3_i;

                    const F neg_i_s3_r = s3_i, neg_i_s3_i = -s3_r;

                    out_0[k] = C(s0_r + s2_r, s0_i + s2_i);
                    out_1[k] = C(s1_r + neg_i_s3_r, s1_i + neg_i_s3_i);
                    out_2[k] = C(s0_r - s2_r, s0_i - s2_i);
                    out_3[k] = C(s1_r - neg_i_s3_r, s1_i - neg_i_s3_i);
                }
            }
        }

        void kernel4(const F* __restrict in_r, const F* __restrict in_i,
                     F* __restrict out_r, F* __restrict out_i, const size_t n) {
            static constexpr F kInvSqrt2 = static_cast<F>(1.0 / std::numbers::sqrt2);
            const size_t half_n = n >> 1;

            for (size_t j = 0; j < half_n; j += 4) {
                const size_t out_idx_a = j << 1;
                const size_t out_idx_b = out_idx_a + 4;
                F ar, ai, br, bi, vr, vi;

                // twiddle = 1
                ar = in_r[j + 0];
                ai = in_i[j + 0];
                br = in_r[j + half_n + 0];
                bi = in_i[j + half_n + 0];
                out_r[out_idx_a + 0] = ar + br;
                out_i[out_idx_a + 0] = ai + bi;
                out_r[out_idx_b + 0] = ar - br;
                out_i[out_idx_b + 0] = ai - bi;

                // twiddle = exp(-i * pi/4)
                ar = in_r[j + 1];
                ai = in_i[j + 1];
                br = in_r[j + half_n + 1];
                bi = in_i[j + half_n + 1];
                vr = (br + bi) * kInvSqrt2;
                vi = (bi - br) * kInvSqrt2;
                out_r[out_idx_a + 1] = ar + vr;
                out_i[out_idx_a + 1] = ai + vi;
                out_r[out_idx_b + 1] = ar - vr;
                out_i[out_idx_b + 1] = ai - vi;

                // twiddle = -i
                ar = in_r[j + 2];
                ai = in_i[j + 2];
                br = in_r[j + half_n + 2];
                bi = in_i[j + half_n + 2];
                vr = bi;
                vi = -br;
                out_r[out_idx_a + 2] = ar + vr;
                out_i[out_idx_a + 2] = ai + vi;
                out_r[out_idx_b + 2] = ar - vr;
                out_i[out_idx_b + 2] = ai - vi;

                // twiddle = exp(-i * 3pi/4)
                ar = in_r[j + 3];
                ai = in_i[j + 3];
                br = in_r[j + half_n + 3];
                bi = in_i[j + half_n + 3];
                vr = (bi - br) * kInvSqrt2;
                vi = (-br - bi) * kInvSqrt2;
                out_r[out_idx_a + 3] = ar + vr;
                out_i[out_idx_a + 3] = ai + vi;
                out_r[out_idx_b + 3] = ar - vr;
                out_i[out_idx_b + 3] = ai - vi;
            }
        }

        static void kernel124_fused(const C* __restrict in,
                                    F* __restrict out_r, F* __restrict out_i, const size_t n) {
            const size_t m = n >> 3; // N / 8
            const hn::ScalableTag<F> d;
            const size_t lanes = hn::Lanes(d);
            size_t j = 0;

            static constexpr F kInvSqrt2 = static_cast<F>(1.0 / std::numbers::sqrt2);
            const auto inv_sqrt2 = hn::Set(d, kInvSqrt2);

            for (; j + lanes <= m; j += lanes) {
                hn::Vec<decltype(d)> x0_r, x0_i, x1_r, x1_i, x2_r, x2_i, x3_r, x3_i;
                hn::Vec<decltype(d)> x4_r, x4_i, x5_r, x5_i, x6_r, x6_i, x7_r, x7_i;

                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + j), x0_r, x0_i);
                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + j + m), x1_r, x1_i);
                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + j + 2 * m), x2_r, x2_i);
                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + j + 3 * m), x3_r, x3_i);
                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + j + 4 * m), x4_r, x4_i);
                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + j + 5 * m), x5_r, x5_i);
                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + j + 6 * m), x6_r, x6_i);
                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + j + 7 * m), x7_r, x7_i);

                auto t0_r = hn::Add(x0_r, x4_r), t0_i = hn::Add(x0_i, x4_i);
                auto t1_r = hn::Sub(x0_r, x4_r), t1_i = hn::Sub(x0_i, x4_i);
                auto t2_r = hn::Add(x2_r, x6_r), t2_i = hn::Add(x2_i, x6_i);
                auto t3_r = hn::Sub(x2_r, x6_r), t3_i = hn::Sub(x2_i, x6_i);

                auto y00_r = hn::Add(t0_r, t2_r), y00_i = hn::Add(t0_i, t2_i);
                auto y01_r = hn::Add(t1_r, t3_i), y01_i = hn::Sub(t1_i, t3_r);
                auto y02_r = hn::Sub(t0_r, t2_r), y02_i = hn::Sub(t0_i, t2_i);
                auto y03_r = hn::Sub(t1_r, t3_i), y03_i = hn::Add(t1_i, t3_r);

                auto u0_r = hn::Add(x1_r, x5_r), u0_i = hn::Add(x1_i, x5_i);
                auto u1_r = hn::Sub(x1_r, x5_r), u1_i = hn::Sub(x1_i, x5_i);
                auto u2_r = hn::Add(x3_r, x7_r), u2_i = hn::Add(x3_i, x7_i);
                auto u3_r = hn::Sub(x3_r, x7_r), u3_i = hn::Sub(x3_i, x7_i);

                auto y10_r = hn::Add(u0_r, u2_r), y10_i = hn::Add(u0_i, u2_i);
                auto y11_r = hn::Add(u1_r, u3_i), y11_i = hn::Sub(u1_i, u3_r);
                auto y12_r = hn::Sub(u0_r, u2_r), y12_i = hn::Sub(u0_i, u2_i);
                auto y13_r = hn::Sub(u1_r, u3_i), y13_i = hn::Add(u1_i, u3_r);

                auto v0_r = y10_r, v0_i = y10_i;
                auto v1_r = hn::Mul(hn::Add(y11_r, y11_i), inv_sqrt2);
                auto v1_i = hn::Mul(hn::Sub(y11_i, y11_r), inv_sqrt2);
                auto v2_r = y12_i, v2_i = hn::Neg(y12_r);
                auto v3_r = hn::Mul(hn::Sub(y13_i, y13_r), inv_sqrt2);
                auto v3_i = hn::Mul(hn::Neg(hn::Add(y13_r, y13_i)), inv_sqrt2);

                auto z00_r = hn::Add(y00_r, v0_r), z00_i = hn::Add(y00_i, v0_i);
                auto z01_r = hn::Add(y01_r, v1_r), z01_i = hn::Add(y01_i, v1_i);
                auto z02_r = hn::Add(y02_r, v2_r), z02_i = hn::Add(y02_i, v2_i);
                auto z03_r = hn::Add(y03_r, v3_r), z03_i = hn::Add(y03_i, v3_i);

                auto z10_r = hn::Sub(y00_r, v0_r), z10_i = hn::Sub(y00_i, v0_i);
                auto z11_r = hn::Sub(y01_r, v1_r), z11_i = hn::Sub(y01_i, v1_i);
                auto z12_r = hn::Sub(y02_r, v2_r), z12_i = hn::Sub(y02_i, v2_i);
                auto z13_r = hn::Sub(y03_r, v3_r), z13_i = hn::Sub(y03_i, v3_i);

                auto lower_r0 = hn::InterleaveLower(d, z00_r, z10_r);
                auto lower_r1 = hn::InterleaveLower(d, z01_r, z11_r);
                auto lower_r2 = hn::InterleaveLower(d, z02_r, z12_r);
                auto lower_r3 = hn::InterleaveLower(d, z03_r, z13_r);

                auto upper_r0 = hn::InterleaveUpper(d, z00_r, z10_r);
                auto upper_r1 = hn::InterleaveUpper(d, z01_r, z11_r);
                auto upper_r2 = hn::InterleaveUpper(d, z02_r, z12_r);
                auto upper_r3 = hn::InterleaveUpper(d, z03_r, z13_r);

                auto lower_i0 = hn::InterleaveLower(d, z00_i, z10_i);
                auto lower_i1 = hn::InterleaveLower(d, z01_i, z11_i);
                auto lower_i2 = hn::InterleaveLower(d, z02_i, z12_i);
                auto lower_i3 = hn::InterleaveLower(d, z03_i, z13_i);

                auto upper_i0 = hn::InterleaveUpper(d, z00_i, z10_i);
                auto upper_i1 = hn::InterleaveUpper(d, z01_i, z11_i);
                auto upper_i2 = hn::InterleaveUpper(d, z02_i, z12_i);
                auto upper_i3 = hn::InterleaveUpper(d, z03_i, z13_i);

                const size_t out_offset = j << 3;
                hn::StoreInterleaved4(lower_r0, lower_r1, lower_r2, lower_r3, d, out_r + out_offset);
                hn::StoreInterleaved4(upper_r0, upper_r1, upper_r2, upper_r3, d, out_r + out_offset + 4 * lanes);

                hn::StoreInterleaved4(lower_i0, lower_i1, lower_i2, lower_i3, d, out_i + out_offset);
                hn::StoreInterleaved4(upper_i0, upper_i1, upper_i2, upper_i3, d, out_i + out_offset + 4 * lanes);
            }

            for (; j < m; ++j) {
                const F x0_r = in[j].real(), x0_i = in[j].imag();
                const F x1_r = in[j + m].real(), x1_i = in[j + m].imag();
                const F x2_r = in[j + 2 * m].real(), x2_i = in[j + 2 * m].imag();
                const F x3_r = in[j + 3 * m].real(), x3_i = in[j + 3 * m].imag();
                const F x4_r = in[j + 4 * m].real(), x4_i = in[j + 4 * m].imag();
                const F x5_r = in[j + 5 * m].real(), x5_i = in[j + 5 * m].imag();
                const F x6_r = in[j + 6 * m].real(), x6_i = in[j + 6 * m].imag();
                const F x7_r = in[j + 7 * m].real(), x7_i = in[j + 7 * m].imag();

                const F t0_r = x0_r + x4_r, t0_i = x0_i + x4_i;
                const F t1_r = x0_r - x4_r, t1_i = x0_i - x4_i;
                const F t2_r = x2_r + x6_r, t2_i = x2_i + x6_i;
                const F t3_r = x2_r - x6_r, t3_i = x2_i - x6_i;

                const F y00_r = t0_r + t2_r, y00_i = t0_i + t2_i;
                const F y01_r = t1_r + t3_i, y01_i = t1_i - t3_r;
                const F y02_r = t0_r - t2_r, y02_i = t0_i - t2_i;
                const F y03_r = t1_r - t3_i, y03_i = t1_i + t3_r;

                const F u0_r = x1_r + x5_r, u0_i = x1_i + x5_i;
                const F u1_r = x1_r - x5_r, u1_i = x1_i - x5_i;
                const F u2_r = x3_r + x7_r, u2_i = x3_i + x7_i;
                const F u3_r = x3_r - x7_r, u3_i = x3_i - x7_i;

                const F y10_r = u0_r + u2_r, y10_i = u0_i + u2_i;
                const F y11_r = u1_r + u3_i, y11_i = u1_i - u3_r;
                const F y12_r = u0_r - u2_r, y12_i = u0_i - u2_i;
                const F y13_r = u1_r - u3_i, y13_i = u1_i + u3_r;

                const F v0_r = y10_r, v0_i = y10_i;
                const F v1_r = (y11_r + y11_i) * kInvSqrt2, v1_i = (y11_i - y11_r) * kInvSqrt2;
                const F v2_r = y12_i, v2_i = -y12_r;
                const F v3_r = (y13_i - y13_r) * kInvSqrt2, v3_i = -(y13_r + y13_i) * kInvSqrt2;

                const size_t out_idx = j << 3;

                out_r[out_idx + 0] = y00_r + v0_r;
                out_i[out_idx + 0] = y00_i + v0_i;
                out_r[out_idx + 1] = y01_r + v1_r;
                out_i[out_idx + 1] = y01_i + v1_i;
                out_r[out_idx + 2] = y02_r + v2_r;
                out_i[out_idx + 2] = y02_i + v2_i;
                out_r[out_idx + 3] = y03_r + v3_r;
                out_i[out_idx + 3] = y03_i + v3_i;

                out_r[out_idx + 4] = y00_r - v0_r;
                out_i[out_idx + 4] = y00_i - v0_i;
                out_r[out_idx + 5] = y01_r - v1_r;
                out_i[out_idx + 5] = y01_i - v1_i;
                out_r[out_idx + 6] = y02_r - v2_r;
                out_i[out_idx + 6] = y02_i - v2_i;
                out_r[out_idx + 7] = y03_r - v3_r;
                out_i[out_idx + 7] = y03_i - v3_i;
            }
        }
    };
}
