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
    class SIMDLowOrderOPT1 {
        using C = std::complex<F>;

    public:
        explicit SIMDLowOrderOPT1(const size_t order) :
            order_(order) {
            if (order < 4) {
                return;
            }
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

            switch (order_) {
            case 0:
                callback_order_0(in_buffer.data(), out_buffer.data());
                return;
            case 1:
                callback_order_1(in_buffer.data(), out_buffer.data());
                return;
            case 2:
                callback_order_2(in_buffer.data(), out_buffer.data());
                return;
            case 3:
                callback_order_3(in_buffer.data(), out_buffer.data());
                return;
            case 4:
                callback_order_4(in_buffer.data(), out_buffer.data());
                return;
            case 5:
                callback_order_5(in_buffer.data(), out_buffer.data());
                return;
            default:
                break;
            }

            const auto n = in_buffer.size();

            F* __restrict in_r = workspace_.get();
            F* __restrict in_i = in_r + stride_;
            F* __restrict out_r = in_i + stride_;
            F* __restrict out_i = out_r + stride_;

            const F* __restrict w_r_ptr = twiddles_r_.get();
            const F* __restrict w_i_ptr = twiddles_i_.get();

            if (order_ % 2 == 1) {
                radix8_first_pass_fused(in_buffer.data(), out_r, out_i, n);
                std::swap(in_r, out_r);
                std::swap(in_i, out_i);
            } else {
                radix4_first_pass_fused(in_buffer.data(), out_r, out_i, n);
                radix4_width4(out_r, out_i, in_r, in_i, n, w_r_ptr, w_i_ptr);
                w_r_ptr += 12;
                w_i_ptr += 12;
            }

            size_t width = (order_ % 2 == 0) ? 16 : 8;

            for (; width < (n >> 2); width <<= 2) {
                radix4(in_r, in_i, out_r, out_i, n, width, w_r_ptr, w_i_ptr);
                std::swap(in_r, out_r);
                std::swap(in_i, out_i);
                w_r_ptr += 3 * width;
                w_i_ptr += 3 * width;
            }
            radix4_last_pass_fused(in_r, in_i, out_buffer.data(), n, width, w_r_ptr, w_i_ptr);
        }

    private:
        size_t order_;
        size_t stride_;
        hwy::AlignedFreeUniquePtr<F[]> twiddles_r_;
        hwy::AlignedFreeUniquePtr<F[]> twiddles_i_;
        hwy::AlignedFreeUniquePtr<F[]> workspace_;

        static void radix4_first_pass_fused(const C* __restrict in,
                                            F* __restrict out_r, F* __restrict out_i, const size_t n) {
            const size_t quarter_n = n >> 2;
            const size_t half_n = n >> 1;
            const size_t three_quarter_n = quarter_n + half_n;

            const hn::ScalableTag<F> d;
            const size_t lanes = hn::Lanes(d);

            for (size_t j = 0; j < quarter_n; j += lanes) {
                hn::Vec<decltype(d)> x0_r, x0_i, x1_r, x1_i, x2_r, x2_i, x3_r, x3_i;

                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + j), x0_r, x0_i);
                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + j + half_n), x2_r, x2_i);

                const auto t0_r = hn::Add(x0_r, x2_r);
                const auto t0_i = hn::Add(x0_i, x2_i);
                const auto t1_r = hn::Sub(x0_r, x2_r);
                const auto t1_i = hn::Sub(x0_i, x2_i);

                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + j + quarter_n), x1_r, x1_i);
                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + j + three_quarter_n), x3_r, x3_i);

                const auto t2_r = hn::Add(x1_r, x3_r);
                const auto t2_i = hn::Add(x1_i, x3_i);
                const auto t3_r = hn::Sub(x1_r, x3_r);
                const auto t3_i = hn::Sub(x1_i, x3_i);

                const auto out0_r = hn::Add(t0_r, t2_r);
                const auto out0_i = hn::Add(t0_i, t2_i);
                const auto out2_r = hn::Sub(t0_r, t2_r);
                const auto out2_i = hn::Sub(t0_i, t2_i);

                const auto t3_neg_i_r = t3_i;
                const auto t3_neg_i_i = hn::Neg(t3_r);

                const auto out1_r = hn::Add(t1_r, t3_neg_i_r);
                const auto out1_i = hn::Add(t1_i, t3_neg_i_i);
                const auto out3_r = hn::Sub(t1_r, t3_neg_i_r);
                const auto out3_i = hn::Sub(t1_i, t3_neg_i_i);

                const size_t out_offset = j << 2;
                hn::StoreInterleaved4(out0_r, out1_r, out2_r, out3_r, d, out_r + out_offset);
                hn::StoreInterleaved4(out0_i, out1_i, out2_i, out3_i, d, out_i + out_offset);
            }
        }

        static void radix8_first_pass_fused(const C* __restrict in,
                                            F* __restrict out_r, F* __restrict out_i, const size_t n) {
            const size_t m = n >> 3;
            const hn::ScalableTag<F> d;
            const size_t lanes = hn::Lanes(d);

            static constexpr F kInvSqrt2 = static_cast<F>(1.0 / std::numbers::sqrt2);
            const auto inv_sqrt2 = hn::Set(d, kInvSqrt2);

            for (size_t j = 0; j + lanes <= m; j += lanes) {
                hn::Vec<decltype(d)> temp_a_r, temp_a_i, temp_b_r, temp_b_i;

                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + j), temp_a_r, temp_a_i);
                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + j + (m << 2)), temp_b_r, temp_b_i);
                const auto t0_r = hn::Add(temp_a_r, temp_b_r), t0_i = hn::Add(temp_a_i, temp_b_i);
                const auto t1_r = hn::Sub(temp_a_r, temp_b_r), t1_i = hn::Sub(temp_a_i, temp_b_i);

                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + j + (m << 1)), temp_a_r, temp_a_i);
                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + j + m * 6), temp_b_r, temp_b_i);
                const auto t2_r = hn::Add(temp_a_r, temp_b_r), t2_i = hn::Add(temp_a_i, temp_b_i);
                const auto t3_r = hn::Sub(temp_a_r, temp_b_r), t3_i = hn::Sub(temp_a_i, temp_b_i);

                const auto y00_r = hn::Add(t0_r, t2_r), y00_i = hn::Add(t0_i, t2_i);
                const auto y01_r = hn::Add(t1_r, t3_i), y01_i = hn::Sub(t1_i, t3_r);
                const auto y02_r = hn::Sub(t0_r, t2_r), y02_i = hn::Sub(t0_i, t2_i);
                const auto y03_r = hn::Sub(t1_r, t3_i), y03_i = hn::Add(t1_i, t3_r);

                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + j + m), temp_a_r, temp_a_i);
                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + j + 5 * m), temp_b_r, temp_b_i);
                const auto u0_r = hn::Add(temp_a_r, temp_b_r), u0_i = hn::Add(temp_a_i, temp_b_i);
                const auto u1_r = hn::Sub(temp_a_r, temp_b_r), u1_i = hn::Sub(temp_a_i, temp_b_i);

                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + j + 3 * m), temp_a_r, temp_a_i);
                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + j + 7 * m), temp_b_r, temp_b_i);
                const auto u2_r = hn::Add(temp_a_r, temp_b_r), u2_i = hn::Add(temp_a_i, temp_b_i);
                const auto u3_r = hn::Sub(temp_a_r, temp_b_r), u3_i = hn::Sub(temp_a_i, temp_b_i);

                const auto y10_r = hn::Add(u0_r, u2_r), y10_i = hn::Add(u0_i, u2_i);
                const auto y11_r = hn::Add(u1_r, u3_i), y11_i = hn::Sub(u1_i, u3_r);
                const auto y12_r = hn::Sub(u0_r, u2_r), y12_i = hn::Sub(u0_i, u2_i);
                const auto y13_r = hn::Sub(u1_r, u3_i), y13_i = hn::Add(u1_i, u3_r);

                const auto v0_r = y10_r, v0_i = y10_i;
                const auto v1_r = hn::Mul(hn::Add(y11_r, y11_i), inv_sqrt2);
                const auto v1_i = hn::Mul(hn::Sub(y11_i, y11_r), inv_sqrt2);
                const auto v2_r = y12_i, v2_i = hn::Neg(y12_r);
                const auto v3_r = hn::Mul(hn::Sub(y13_i, y13_r), inv_sqrt2);
                const auto v3_i = hn::Mul(hn::Neg(hn::Add(y13_r, y13_i)), inv_sqrt2);

                const auto z00_r = hn::Add(y00_r, v0_r), z00_i = hn::Add(y00_i, v0_i);
                const auto z01_r = hn::Add(y01_r, v1_r), z01_i = hn::Add(y01_i, v1_i);
                const auto z02_r = hn::Add(y02_r, v2_r), z02_i = hn::Add(y02_i, v2_i);
                const auto z03_r = hn::Add(y03_r, v3_r), z03_i = hn::Add(y03_i, v3_i);

                const auto z10_r = hn::Sub(y00_r, v0_r), z10_i = hn::Sub(y00_i, v0_i);
                const auto z11_r = hn::Sub(y01_r, v1_r), z11_i = hn::Sub(y01_i, v1_i);
                const auto z12_r = hn::Sub(y02_r, v2_r), z12_i = hn::Sub(y02_i, v2_i);
                const auto z13_r = hn::Sub(y03_r, v3_r), z13_i = hn::Sub(y03_i, v3_i);

                const auto lower_r0 = hn::InterleaveLower(d, z00_r, z10_r);
                const auto lower_r1 = hn::InterleaveLower(d, z01_r, z11_r);
                const auto lower_r2 = hn::InterleaveLower(d, z02_r, z12_r);
                const auto lower_r3 = hn::InterleaveLower(d, z03_r, z13_r);

                const auto upper_r0 = hn::InterleaveUpper(d, z00_r, z10_r);
                const auto upper_r1 = hn::InterleaveUpper(d, z01_r, z11_r);
                const auto upper_r2 = hn::InterleaveUpper(d, z02_r, z12_r);
                const auto upper_r3 = hn::InterleaveUpper(d, z03_r, z13_r);

                const auto lower_i0 = hn::InterleaveLower(d, z00_i, z10_i);
                const auto lower_i1 = hn::InterleaveLower(d, z01_i, z11_i);
                const auto lower_i2 = hn::InterleaveLower(d, z02_i, z12_i);
                const auto lower_i3 = hn::InterleaveLower(d, z03_i, z13_i);

                const auto upper_i0 = hn::InterleaveUpper(d, z00_i, z10_i);
                const auto upper_i1 = hn::InterleaveUpper(d, z01_i, z11_i);
                const auto upper_i2 = hn::InterleaveUpper(d, z02_i, z12_i);
                const auto upper_i3 = hn::InterleaveUpper(d, z03_i, z13_i);

                const size_t out_offset = j << 3;
                hn::StoreInterleaved4(lower_r0, lower_r1, lower_r2, lower_r3, d, out_r + out_offset);
                hn::StoreInterleaved4(upper_r0, upper_r1, upper_r2, upper_r3, d, out_r + out_offset + (lanes << 2));

                hn::StoreInterleaved4(lower_i0, lower_i1, lower_i2, lower_i3, d, out_i + out_offset);
                hn::StoreInterleaved4(upper_i0, upper_i1, upper_i2, upper_i3, d, out_i + out_offset + (lanes << 2));
            }
        }

        void radix4_width4(const F* __restrict in_r, const F* __restrict in_i,
                           F* __restrict out_r, F* __restrict out_i,
                           const size_t n,
                           const F* __restrict w_r_ptr, const F* __restrict w_i_ptr) {
            const size_t quarter_n = n >> 2;
            const size_t half_n = n >> 1;
            const size_t three_quarter_n = quarter_n * 3;

            const hn::FixedTag<F, 4> d;

            const auto w1_r = hn::LoadU(d, w_r_ptr);
            const auto w1_i = hn::LoadU(d, w_i_ptr);
            const auto w2_r = hn::LoadU(d, w_r_ptr + 4);
            const auto w2_i = hn::LoadU(d, w_i_ptr + 4);
            const auto w3_r = hn::LoadU(d, w_r_ptr + 8);
            const auto w3_i = hn::LoadU(d, w_i_ptr + 8);

            for (size_t j = 0; j < quarter_n; j += 4) {
                const auto i1 = hn::LoadU(d, in_i + j + quarter_n);
                const auto r1 = hn::LoadU(d, in_r + j + quarter_n);
                const auto t1_r = hn::NegMulAdd(i1, w1_i, hn::Mul(r1, w1_r));
                const auto t1_i = hn::MulAdd(i1, w1_r, hn::Mul(r1, w1_i));

                const auto i3 = hn::LoadU(d, in_i + j + three_quarter_n);
                const auto r3 = hn::LoadU(d, in_r + j + three_quarter_n);
                const auto t3_r = hn::NegMulAdd(i3, w3_i, hn::Mul(r3, w3_r));
                const auto t3_i = hn::MulAdd(i3, w3_r, hn::Mul(r3, w3_i));

                const auto s2_r = hn::Add(t1_r, t3_r);
                const auto s2_i = hn::Add(t1_i, t3_i);
                const auto s3_r = hn::Sub(t1_r, t3_r);
                const auto s3_i = hn::Sub(t1_i, t3_i);

                const auto i2 = hn::LoadU(d, in_i + j + half_n);
                const auto r2 = hn::LoadU(d, in_r + j + half_n);
                const auto t2_r = hn::NegMulAdd(i2, w2_i, hn::Mul(r2, w2_r));
                const auto t2_i = hn::MulAdd(i2, w2_r, hn::Mul(r2, w2_i));

                const auto r0 = hn::LoadU(d, in_r + j);
                const auto i0 = hn::LoadU(d, in_i + j);

                const auto s0_r = hn::Add(r0, t2_r);
                const auto s0_i = hn::Add(i0, t2_i);
                const auto s1_r = hn::Sub(r0, t2_r);
                const auto s1_i = hn::Sub(i0, t2_i);

                hn::StoreU(hn::Add(s0_r, s2_r), d, out_r + (j << 2));
                hn::StoreU(hn::Add(s0_i, s2_i), d, out_i + (j << 2));
                hn::StoreU(hn::Sub(s0_r, s2_r), d, out_r + (j << 2) + 8);
                hn::StoreU(hn::Sub(s0_i, s2_i), d, out_i + (j << 2) + 8);

                hn::StoreU(hn::Add(s1_r, s3_i), d, out_r + (j << 2) + 4);
                hn::StoreU(hn::Sub(s1_i, s3_r), d, out_i + (j << 2) + 4);
                hn::StoreU(hn::Sub(s1_r, s3_i), d, out_r + (j << 2) + 12);
                hn::StoreU(hn::Add(s1_i, s3_r), d, out_i + (j << 2) + 12);
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

            const size_t mask = width - 1;

            for (size_t i = 0; i < quarter_n; i += lanes) {
                const auto r0 = hn::LoadU(d, in_r + i);
                const auto i0 = hn::LoadU(d, in_i + i);
                const auto r1 = hn::LoadU(d, in_r + quarter_n + i);
                const auto i1 = hn::LoadU(d, in_i + quarter_n + i);
                const auto r2 = hn::LoadU(d, in_r + half_n + i);
                const auto i2 = hn::LoadU(d, in_i + half_n + i);
                const auto r3 = hn::LoadU(d, in_r + three_quarter_n + i);
                const auto i3 = hn::LoadU(d, in_i + three_quarter_n + i);

                const size_t k = i & mask;
                const auto w1_r = hn::LoadU(d, w_r_ptr + k);
                const auto w1_i = hn::LoadU(d, w_i_ptr + k);
                const auto w2_r = hn::LoadU(d, w_r_ptr + width + k);
                const auto w2_i = hn::LoadU(d, w_i_ptr + width + k);
                const auto w3_r = hn::LoadU(d, w_r_ptr + (width << 1) + k);
                const auto w3_i = hn::LoadU(d, w_i_ptr + (width << 1) + k);

                const auto t1_r = hn::NegMulAdd(i1, w1_i, hn::Mul(r1, w1_r));
                const auto t1_i = hn::MulAdd(i1, w1_r, hn::Mul(r1, w1_i));
                const auto t3_r = hn::NegMulAdd(i3, w3_i, hn::Mul(r3, w3_r));
                const auto t3_i = hn::MulAdd(i3, w3_r, hn::Mul(r3, w3_i));

                const auto s2_r = hn::Add(t1_r, t3_r);
                const auto s2_i = hn::Add(t1_i, t3_i);
                const auto s3_r = hn::Sub(t1_r, t3_r);
                const auto s3_i = hn::Sub(t1_i, t3_i);

                const auto t2_r = hn::NegMulAdd(i2, w2_i, hn::Mul(r2, w2_r));
                const auto t2_i = hn::MulAdd(i2, w2_r, hn::Mul(r2, w2_i));

                const auto s0_r = hn::Add(r0, t2_r);
                const auto s0_i = hn::Add(i0, t2_i);
                const auto s1_r = hn::Sub(r0, t2_r);
                const auto s1_i = hn::Sub(i0, t2_i);

                const size_t j_times_4 = (i & ~mask) << 2;
                const size_t out_idx = j_times_4 + k;

                hn::StoreU(hn::Add(s0_r, s2_r), d, out_r + out_idx);
                hn::StoreU(hn::Add(s0_i, s2_i), d, out_i + out_idx);

                hn::StoreU(hn::Add(s1_r, s3_i), d, out_r + out_idx + width);
                hn::StoreU(hn::Sub(s1_i, s3_r), d, out_i + out_idx + width);

                hn::StoreU(hn::Sub(s0_r, s2_r), d, out_r + out_idx + (width << 1));
                hn::StoreU(hn::Sub(s0_i, s2_i), d, out_i + out_idx + (width << 1));

                hn::StoreU(hn::Sub(s1_r, s3_i), d, out_r + out_idx + width * 3);
                hn::StoreU(hn::Add(s1_i, s3_r), d, out_i + out_idx + width * 3);
            }
        }

        static void radix4_last_pass_fused(const F* __restrict in_r, const F* __restrict in_i,
                                           C* __restrict out,
                                           const size_t n, const size_t width,
                                           const F* __restrict w_r_ptr, const F* __restrict w_i_ptr) {
            const size_t quarter_n = n >> 2;
            const size_t half_n = n >> 1;
            const size_t three_quarter_n = quarter_n * 3;

            const hn::ScalableTag<F> d;
            const size_t lanes = hn::Lanes(d);

            const size_t mask = width - 1;

            for (size_t i = 0; i < quarter_n; i += lanes) {
                const auto r0 = hn::LoadU(d, in_r + i);
                const auto i0 = hn::LoadU(d, in_i + i);
                const auto r1 = hn::LoadU(d, in_r + quarter_n + i);
                const auto i1 = hn::LoadU(d, in_i + quarter_n + i);
                const auto r2 = hn::LoadU(d, in_r + half_n + i);
                const auto i2 = hn::LoadU(d, in_i + half_n + i);
                const auto r3 = hn::LoadU(d, in_r + three_quarter_n + i);
                const auto i3 = hn::LoadU(d, in_i + three_quarter_n + i);

                const size_t k = i & mask;
                const auto w1_r = hn::LoadU(d, w_r_ptr + k);
                const auto w1_i = hn::LoadU(d, w_i_ptr + k);
                const auto w2_r = hn::LoadU(d, w_r_ptr + width + k);
                const auto w2_i = hn::LoadU(d, w_i_ptr + width + k);
                const auto w3_r = hn::LoadU(d, w_r_ptr + (width << 1) + k);
                const auto w3_i = hn::LoadU(d, w_i_ptr + (width << 1) + k);

                const auto t1_r = hn::NegMulAdd(i1, w1_i, hn::Mul(r1, w1_r));
                const auto t1_i = hn::MulAdd(i1, w1_r, hn::Mul(r1, w1_i));
                const auto t3_r = hn::NegMulAdd(i3, w3_i, hn::Mul(r3, w3_r));
                const auto t3_i = hn::MulAdd(i3, w3_r, hn::Mul(r3, w3_i));

                const auto s2_r = hn::Add(t1_r, t3_r);
                const auto s2_i = hn::Add(t1_i, t3_i);
                const auto s3_r = hn::Sub(t1_r, t3_r);
                const auto s3_i = hn::Sub(t1_i, t3_i);

                const auto t2_r = hn::NegMulAdd(i2, w2_i, hn::Mul(r2, w2_r));
                const auto t2_i = hn::MulAdd(i2, w2_r, hn::Mul(r2, w2_i));

                const auto s0_r = hn::Add(r0, t2_r);
                const auto s0_i = hn::Add(i0, t2_i);
                const auto s1_r = hn::Sub(r0, t2_r);
                const auto s1_i = hn::Sub(i0, t2_i);

                const size_t j_times_4 = (i & ~mask) << 2;
                const size_t out_idx = j_times_4 + k;

                hn::StoreInterleaved2(hn::Add(s0_r, s2_r), hn::Add(s0_i, s2_i), d,
                                      reinterpret_cast<F*>(out + out_idx));
                hn::StoreInterleaved2(hn::Add(s1_r, s3_i), hn::Sub(s1_i, s3_r), d,
                                      reinterpret_cast<F*>(out + out_idx + width));
                hn::StoreInterleaved2(hn::Sub(s0_r, s2_r), hn::Sub(s0_i, s2_i), d,
                                      reinterpret_cast<F*>(out + out_idx + (width << 1)));
                hn::StoreInterleaved2(hn::Sub(s1_r, s3_i), hn::Add(s1_i, s3_r), d,
                                      reinterpret_cast<F*>(out + out_idx + width * 3));
            }
        }

        void callback_order_0(const C* __restrict in, C* __restrict out) const {
            out[0] = in[0];
        }

        void callback_order_1(const C* __restrict in, C* __restrict out) const {
            out[0] = in[0] + in[1];
            out[1] = in[0] - in[1];
        }

        void callback_order_2(const C* __restrict in, C* __restrict out) const {
            const C t0 = in[0] + in[2];
            const C t1 = in[0] - in[2];
            const C t2 = in[1] + in[3];
            const C t3 = in[1] - in[3];
            out[0] = t0 + t2;
            out[2] = t0 - t2;
            out[1] = t1 + C(t3.imag(), -t3.real());
            out[3] = t1 - C(t3.imag(), -t3.real());
        }

        void callback_order_3(const C* __restrict in, C* __restrict out) const {
            static constexpr F kInvSqrt2 = static_cast<F>(1.0 / std::numbers::sqrt2);
            const F x0_r = in[0].real(), x0_i = in[0].imag();
            const F x1_r = in[1].real(), x1_i = in[1].imag();
            const F x2_r = in[2].real(), x2_i = in[2].imag();
            const F x3_r = in[3].real(), x3_i = in[3].imag();
            const F x4_r = in[4].real(), x4_i = in[4].imag();
            const F x5_r = in[5].real(), x5_i = in[5].imag();
            const F x6_r = in[6].real(), x6_i = in[6].imag();
            const F x7_r = in[7].real(), x7_i = in[7].imag();

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

            out[0] = C(y00_r + v0_r, y00_i + v0_i);
            out[1] = C(y01_r + v1_r, y01_i + v1_i);
            out[2] = C(y02_r + v2_r, y02_i + v2_i);
            out[3] = C(y03_r + v3_r, y03_i + v3_i);
            out[4] = C(y00_r - v0_r, y00_i - v0_i);
            out[5] = C(y01_r - v1_r, y01_i - v1_i);
            out[6] = C(y02_r - v2_r, y02_i - v2_i);
            out[7] = C(y03_r - v3_r, y03_i - v3_i);
        }

        void callback_order_4(const C* __restrict in, C* __restrict out) const {
            const hn::FixedTag<F, 4> d;
            alignas(64) F tmp_r[16];
            alignas(64) F tmp_i[16];

            {
                hn::Vec<decltype(d)> x0_r, x0_i, x1_r, x1_i, x2_r, x2_i, x3_r, x3_i;
                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in), x0_r, x0_i);
                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + 8), x2_r, x2_i);

                const auto t0_r = hn::Add(x0_r, x2_r), t0_i = hn::Add(x0_i, x2_i);
                const auto t1_r = hn::Sub(x0_r, x2_r), t1_i = hn::Sub(x0_i, x2_i);

                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + 4), x1_r, x1_i);
                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + 12), x3_r, x3_i);

                const auto t2_r = hn::Add(x1_r, x3_r), t2_i = hn::Add(x1_i, x3_i);
                const auto t3_r = hn::Sub(x1_r, x3_r), t3_i = hn::Sub(x1_i, x3_i);

                const auto out0_r = hn::Add(t0_r, t2_r), out0_i = hn::Add(t0_i, t2_i);
                const auto out2_r = hn::Sub(t0_r, t2_r), out2_i = hn::Sub(t0_i, t2_i);
                const auto out1_r = hn::Add(t1_r, t3_i), out1_i = hn::Sub(t1_i, t3_r);
                const auto out3_r = hn::Sub(t1_r, t3_i), out3_i = hn::Add(t1_i, t3_r);

                hn::StoreInterleaved4(out0_r, out1_r, out2_r, out3_r, d, tmp_r);
                hn::StoreInterleaved4(out0_i, out1_i, out2_i, out3_i, d, tmp_i);
            }

            {
                const F* w_r_base = twiddles_r_.get();
                const F* w_i_base = twiddles_i_.get();

                const auto i1 = hn::Load(d, tmp_i + 4), r1 = hn::Load(d, tmp_r + 4);
                const auto w1_r = hn::Load(d, w_r_base), w1_i = hn::Load(d, w_i_base);
                const auto t1_r = hn::NegMulAdd(i1, w1_i, hn::Mul(r1, w1_r));
                const auto t1_i = hn::MulAdd(i1, w1_r, hn::Mul(r1, w1_i));

                const auto i3 = hn::Load(d, tmp_i + 12), r3 = hn::Load(d, tmp_r + 12);
                const auto w3_r = hn::Load(d, w_r_base + 8), w3_i = hn::Load(d, w_i_base + 8);
                const auto t3_r = hn::NegMulAdd(i3, w3_i, hn::Mul(r3, w3_r));
                const auto t3_i = hn::MulAdd(i3, w3_r, hn::Mul(r3, w3_i));

                const auto s2_r = hn::Add(t1_r, t3_r), s2_i = hn::Add(t1_i, t3_i);
                const auto s3_r = hn::Sub(t1_r, t3_r), s3_i = hn::Sub(t1_i, t3_i);

                const auto i2 = hn::Load(d, tmp_i + 8), r2 = hn::Load(d, tmp_r + 8);
                const auto w2_r = hn::Load(d, w_r_base + 4), w2_i = hn::Load(d, w_i_base + 4);
                const auto t2_r = hn::NegMulAdd(i2, w2_i, hn::Mul(r2, w2_r));
                const auto t2_i = hn::MulAdd(i2, w2_r, hn::Mul(r2, w2_i));

                const auto r0 = hn::Load(d, tmp_r), i0 = hn::Load(d, tmp_i);
                const auto s0_r = hn::Add(r0, t2_r), s0_i = hn::Add(i0, t2_i);
                const auto s1_r = hn::Sub(r0, t2_r), s1_i = hn::Sub(i0, t2_i);

                hn::StoreInterleaved2(hn::Add(s0_r, s2_r), hn::Add(s0_i, s2_i), d, reinterpret_cast<F*>(out));
                hn::StoreInterleaved2(hn::Add(s1_r, s3_i), hn::Sub(s1_i, s3_r), d, reinterpret_cast<F*>(out + 4));
                hn::StoreInterleaved2(hn::Sub(s0_r, s2_r), hn::Sub(s0_i, s2_i), d, reinterpret_cast<F*>(out + 8));
                hn::StoreInterleaved2(hn::Sub(s1_r, s3_i), hn::Add(s1_i, s3_r), d, reinterpret_cast<F*>(out + 12));
            }
        }

        void callback_order_5(const C* __restrict in, C* __restrict out) const {
            const hn::FixedTag<F, 4> d;
            alignas(64) F tmp_r[32];
            alignas(64) F tmp_i[32];

            {
                const F kInvSqrt2 = static_cast<F>(1.0 / std::numbers::sqrt2);
                const auto inv_sqrt2 = hn::Set(d, kInvSqrt2);
                hn::Vec<decltype(d)> temp_a_r, temp_a_i, temp_b_r, temp_b_i;

                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in), temp_a_r, temp_a_i);
                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + 16), temp_b_r, temp_b_i);
                auto t0_r = hn::Add(temp_a_r, temp_b_r), t0_i = hn::Add(temp_a_i, temp_b_i);
                auto t1_r = hn::Sub(temp_a_r, temp_b_r), t1_i = hn::Sub(temp_a_i, temp_b_i);

                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + 8), temp_a_r, temp_a_i);
                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + 24), temp_b_r, temp_b_i);
                auto t2_r = hn::Add(temp_a_r, temp_b_r), t2_i = hn::Add(temp_a_i, temp_b_i);
                auto t3_r = hn::Sub(temp_a_r, temp_b_r), t3_i = hn::Sub(temp_a_i, temp_b_i);

                auto y00_r = hn::Add(t0_r, t2_r), y00_i = hn::Add(t0_i, t2_i);
                auto y01_r = hn::Add(t1_r, t3_i), y01_i = hn::Sub(t1_i, t3_r);
                auto y02_r = hn::Sub(t0_r, t2_r), y02_i = hn::Sub(t0_i, t2_i);
                auto y03_r = hn::Sub(t1_r, t3_i), y03_i = hn::Add(t1_i, t3_r);

                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + 4), temp_a_r, temp_a_i);
                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + 20), temp_b_r, temp_b_i);
                auto u0_r = hn::Add(temp_a_r, temp_b_r), u0_i = hn::Add(temp_a_i, temp_b_i);
                auto u1_r = hn::Sub(temp_a_r, temp_b_r), u1_i = hn::Sub(temp_a_i, temp_b_i);

                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + 12), temp_a_r, temp_a_i);
                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + 28), temp_b_r, temp_b_i);
                auto u2_r = hn::Add(temp_a_r, temp_b_r), u2_i = hn::Add(temp_a_i, temp_b_i);
                auto u3_r = hn::Sub(temp_a_r, temp_b_r), u3_i = hn::Sub(temp_a_i, temp_b_i);

                auto y10_r = hn::Add(u0_r, u2_r), y10_i = hn::Add(u0_i, u2_i);
                auto y11_r = hn::Add(u1_r, u3_i), y11_i = hn::Sub(u1_i, u3_r);
                auto y12_r = hn::Sub(u0_r, u2_r), y12_i = hn::Sub(u0_i, u2_i);
                auto y13_r = hn::Sub(u1_r, u3_i), y13_i = hn::Add(u1_i, u3_r);

                auto v0_r = y10_r, v0_i = y10_i;
                auto v1_r = hn::Mul(hn::Add(y11_r, y11_i), inv_sqrt2), v1_i = hn::Mul(hn::Sub(y11_i, y11_r), inv_sqrt2);
                auto v2_r = y12_i, v2_i = hn::Neg(y12_r);
                auto v3_r = hn::Mul(hn::Sub(y13_i, y13_r), inv_sqrt2), v3_i = hn::Mul(
                    hn::Neg(hn::Add(y13_r, y13_i)), inv_sqrt2);

                auto z00_r = hn::Add(y00_r, v0_r), z00_i = hn::Add(y00_i, v0_i);
                auto z01_r = hn::Add(y01_r, v1_r), z01_i = hn::Add(y01_i, v1_i);
                auto z02_r = hn::Add(y02_r, v2_r), z02_i = hn::Add(y02_i, v2_i);
                auto z03_r = hn::Add(y03_r, v3_r), z03_i = hn::Add(y03_i, v3_i);

                auto z10_r = hn::Sub(y00_r, v0_r), z10_i = hn::Sub(y00_i, v0_i);
                auto z11_r = hn::Sub(y01_r, v1_r), z11_i = hn::Sub(y01_i, v1_i);
                auto z12_r = hn::Sub(y02_r, v2_r), z12_i = hn::Sub(y02_i, v2_i);
                auto z13_r = hn::Sub(y03_r, v3_r), z13_i = hn::Sub(y03_i, v3_i);

                auto lower_r0 = hn::InterleaveLower(d, z00_r, z10_r), upper_r0 = hn::InterleaveUpper(d, z00_r, z10_r);
                auto lower_r1 = hn::InterleaveLower(d, z01_r, z11_r), upper_r1 = hn::InterleaveUpper(d, z01_r, z11_r);
                auto lower_r2 = hn::InterleaveLower(d, z02_r, z12_r), upper_r2 = hn::InterleaveUpper(d, z02_r, z12_r);
                auto lower_r3 = hn::InterleaveLower(d, z03_r, z13_r), upper_r3 = hn::InterleaveUpper(d, z03_r, z13_r);

                auto lower_i0 = hn::InterleaveLower(d, z00_i, z10_i), upper_i0 = hn::InterleaveUpper(d, z00_i, z10_i);
                auto lower_i1 = hn::InterleaveLower(d, z01_i, z11_i), upper_i1 = hn::InterleaveUpper(d, z01_i, z11_i);
                auto lower_i2 = hn::InterleaveLower(d, z02_i, z12_i), upper_i2 = hn::InterleaveUpper(d, z02_i, z12_i);
                auto lower_i3 = hn::InterleaveLower(d, z03_i, z13_i), upper_i3 = hn::InterleaveUpper(d, z03_i, z13_i);

                hn::StoreInterleaved4(lower_r0, lower_r1, lower_r2, lower_r3, d, tmp_r);
                hn::StoreInterleaved4(upper_r0, upper_r1, upper_r2, upper_r3, d, tmp_r + 16);
                hn::StoreInterleaved4(lower_i0, lower_i1, lower_i2, lower_i3, d, tmp_i);
                hn::StoreInterleaved4(upper_i0, upper_i1, upper_i2, upper_i3, d, tmp_i + 16);
            }

            {
                const F* w1_r_base = twiddles_r_.get();
                const F* w1_i_base = twiddles_i_.get();
                const F* w2_r_base = w1_r_base + 8;
                const F* w2_i_base = w1_i_base + 8;
                const F* w3_r_base = w1_r_base + 16;
                const F* w3_i_base = w1_i_base + 16;

                for (size_t k = 0; k < 8; k += 4) {
                    const auto i1 = hn::Load(d, tmp_i + 8 + k), r1 = hn::Load(d, tmp_r + 8 + k);
                    const auto w1_r = hn::Load(d, w1_r_base + k), w1_i = hn::Load(d, w1_i_base + k);
                    const auto t1_r = hn::NegMulAdd(i1, w1_i, hn::Mul(r1, w1_r));
                    const auto t1_i = hn::MulAdd(i1, w1_r, hn::Mul(r1, w1_i));

                    const auto i3 = hn::Load(d, tmp_i + 24 + k), r3 = hn::Load(d, tmp_r + 24 + k);
                    const auto w3_r = hn::Load(d, w3_r_base + k), w3_i = hn::Load(d, w3_i_base + k);
                    const auto t3_r = hn::NegMulAdd(i3, w3_i, hn::Mul(r3, w3_r));
                    const auto t3_i = hn::MulAdd(i3, w3_r, hn::Mul(r3, w3_i));

                    const auto s2_r = hn::Add(t1_r, t3_r), s2_i = hn::Add(t1_i, t3_i);
                    const auto s3_r = hn::Sub(t1_r, t3_r), s3_i = hn::Sub(t1_i, t3_i);

                    const auto i2 = hn::Load(d, tmp_i + 16 + k), r2 = hn::Load(d, tmp_r + 16 + k);
                    const auto w2_r = hn::Load(d, w2_r_base + k), w2_i = hn::Load(d, w2_i_base + k);
                    const auto t2_r = hn::NegMulAdd(i2, w2_i, hn::Mul(r2, w2_r));
                    const auto t2_i = hn::MulAdd(i2, w2_r, hn::Mul(r2, w2_i));

                    const auto r0 = hn::Load(d, tmp_r + k), i0 = hn::Load(d, tmp_i + k);
                    const auto s0_r = hn::Add(r0, t2_r), s0_i = hn::Add(i0, t2_i);
                    const auto s1_r = hn::Sub(r0, t2_r), s1_i = hn::Sub(i0, t2_i);

                    hn::StoreInterleaved2(hn::Add(s0_r, s2_r), hn::Add(s0_i, s2_i), d, reinterpret_cast<F*>(out + k));
                    hn::StoreInterleaved2(hn::Add(s1_r, s3_i), hn::Sub(s1_i, s3_r), d,
                                          reinterpret_cast<F*>(out + 8 + k));
                    hn::StoreInterleaved2(hn::Sub(s0_r, s2_r), hn::Sub(s0_i, s2_i), d,
                                          reinterpret_cast<F*>(out + 16 + k));
                    hn::StoreInterleaved2(hn::Sub(s1_r, s3_i), hn::Add(s1_i, s3_r), d,
                                          reinterpret_cast<F*>(out + 24 + k));
                }
            }
        }
    };
}
