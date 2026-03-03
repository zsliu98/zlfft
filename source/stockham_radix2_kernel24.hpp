#pragma once

#include <vector>
#include <span>
#include <complex>
#include <cmath>
#include <algorithm>
#include <numbers>
#include <cassert>

namespace zlfft {
    template <typename F>
    class StockhamRadix2Kernel24 {
        using C = std::complex<F>;

    public:
        explicit StockhamRadix2Kernel24(const size_t order) {
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

            kernel12(in, out, n);
            kernel4(out, in, n);
            const C* __restrict w_ptr = twiddles_.data() + 7;
            for (size_t half = 8; half < n; half <<= 1) {
                for (size_t j = 0; j < (n >> 1); j += half) {
                    const C* __restrict group_in_a = &in[j];
                    const C* __restrict group_in_b = &in[j + (n >> 1)];
                    C* __restrict group_out_a = &out[j << 1];
                    C* __restrict group_out_b = &out[(j << 1) + half];
                    for (size_t k = 0; k < half; ++k) {
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
            const C* __restrict group_in_0 = &in[0];
            const C* __restrict group_in_1 = &in[quarter_n];
            const C* __restrict group_in_2 = &in[half_n];
            const C* __restrict group_in_3 = &in[three_quarter_n];

            for (size_t j = 0; j < quarter_n; ++j) {
                const C x0 = group_in_0[j];
                const C x1 = group_in_1[j];
                const C x2 = group_in_2[j];
                const C x3 = group_in_3[j];
                const C t0 = x0 + x2;
                const C t1 = x0 - x2;
                const C t2 = x1 + x3;
                const C t3 = x1 - x3;
                const C t3_neg_i = { t3.imag(), -t3.real() };
                C* __restrict out_ptr = &out[j << 2];
                out_ptr[0] = t0 + t2;
                out_ptr[1] = t1 + t3_neg_i;
                out_ptr[2] = t0 - t2;
                out_ptr[3] = t1 - t3_neg_i;
            }
        }

        void kernel4(C* __restrict in, C* __restrict out, const size_t n) {
            static constexpr F kInvSqrt2 = static_cast<F>(1.0 / std::numbers::sqrt2);
            for (size_t j = 0; j < (n >> 1); j += 4) {
                const C* __restrict group_in_a = &in[j];
                const C* __restrict group_in_b = &in[j + (n >> 1)];
                C* __restrict group_out_a = &out[j << 1];
                C* __restrict group_out_b = &out[(j << 1) + 4];
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
