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
    class NaiveStockhamRadix4 {
        using C = std::complex<F>;

    public:
        explicit NaiveStockhamRadix4(const size_t order) {
            const auto n = static_cast<size_t>(1) << order;
            twiddles_.reserve(n);

            for (size_t m = 1; m < n; m <<= 2) {
                const double angle_step = -2.0 * std::numbers::pi / static_cast<double>(m << 2);

                for (size_t k = 0; k < m; ++k) {
                    const double angle = static_cast<double>(k) * angle_step;
                    const auto w1 = C(std::cos(angle), std::sin(angle));
                    const auto w2 = w1 * w1;
                    const auto w3 = w2 * w1;

                    twiddles_.emplace_back(w1);
                    twiddles_.emplace_back(w2);
                    twiddles_.emplace_back(w3);
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

            const size_t quarter_n = n >> 2;

            for (size_t half = 1; half < n; half <<= 2) {
                const C* __restrict group_in_0 = &in[0];
                const C* __restrict group_in_1 = &in[n >> 2];
                const C* __restrict group_in_2 = &in[n >> 1];
                const C* __restrict group_in_3 = &in[(n >> 2) + (n >> 1)];
                C* __restrict group_out_0 = &out[0];
                C* __restrict group_out_1 = &out[half];
                C* __restrict group_out_2 = &out[half << 1];
                C* __restrict group_out_3 = &out[half + (half << 1)];
                for (size_t j = 0; j < quarter_n; j += half) {
                    const C* __restrict in_0 = &group_in_0[j];
                    const C* __restrict in_1 = &group_in_1[j];
                    const C* __restrict in_2 = &group_in_2[j];
                    const C* __restrict in_3 = &group_in_3[j];

                    const size_t out_shift = j << 2;
                    C* __restrict out_0 = &group_out_0[out_shift];
                    C* __restrict out_1 = &group_out_1[out_shift];
                    C* __restrict out_2 = &group_out_2[out_shift];
                    C* __restrict out_3 = &group_out_3[out_shift];
                    const C* __restrict local_w_ptr = w_ptr;

                    for (size_t k = 0; k < half; ++k) {
                        const auto x0 = in_0[k];
                        const auto x1 = in_1[k];
                        const auto x2 = in_2[k];
                        const auto x3 = in_3[k];

                        const auto w1 = local_w_ptr[0];
                        const auto w2 = local_w_ptr[1];
                        const auto w3 = local_w_ptr[2];
                        local_w_ptr += 3;

                        const auto v1 = x1 * w1;
                        const auto v2 = x2 * w2;
                        const auto v3 = x3 * w3;

                        const auto t0 = x0 + v2;
                        const auto t1 = x0 - v2;
                        const auto t2 = v1 + v3;
                        const auto t3 = C(v1.imag() - v3.imag(), v3.real() - v1.real());

                        out_0[k] = t0 + t2;
                        out_1[k] = t1 + t3;
                        out_2[k] = t0 - t2;
                        out_3[k] = t1 - t3;
                    }
                }
                w_ptr += 3 * half;
                std::swap(in, out);
            }

            if (in == in_buffer.data()) {
                std::copy(in_buffer.begin(), in_buffer.end(), out_buffer.begin());
            }
        }

    private:
        std::vector<C> twiddles_;
    };
}
