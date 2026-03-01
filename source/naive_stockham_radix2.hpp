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
    class NaiveStockhamRadix2 {
        using C = std::complex<F>;

    public:
        explicit NaiveStockhamRadix2(const size_t order) {
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
            for (size_t half = 1; half < n; half <<= 1) {
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
    };
}
