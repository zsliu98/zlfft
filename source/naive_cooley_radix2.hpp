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
    class NaiveCooleyRadix2 {
        using C = std::complex<F>;

    public:
        explicit NaiveCooleyRadix2(const size_t order) {
            const auto n = static_cast<size_t>(1) << order;
            twiddles_.reserve(n / 2);
            const double angle_step = -2.0 * std::numbers::pi / static_cast<double>(n);
            for (size_t k = 0; k < n / 2; ++k) {
                const auto angle = static_cast<double>(k) * angle_step;
                twiddles_.emplace_back(static_cast<F>(std::cos(angle)),
                                       static_cast<F>(std::sin(angle)));
            }
        }

        void forward(std::span<C> in_buffer, std::span<C> out_buffer) {
            // in/out buffer size should match the init size
            assert(in_buffer.size() == (twiddles_.size() << 1));
            // in/out buffer should have the same size
            assert(in_buffer.size() == out_buffer.size());
            if (in_buffer.data() != out_buffer.data()) {
                std::copy(in_buffer.begin(), in_buffer.end(), out_buffer.begin());
            }

            const size_t n = out_buffer.size();
            C* __restrict data = out_buffer.data();

            // bit-reversal
            size_t j = 0;
            for (size_t i = 1; i < n; ++i) {
                size_t bit = n >> 1;
                while (j & bit) {
                    j ^= bit;
                    bit >>= 1;
                }
                j ^= bit;
                if (i < j) {
                    std::swap(data[i], data[j]);
                }
            }

            const C* __restrict w_ptr = twiddles_.data();
            for (size_t half = 1; half < n; half <<= 1) {
                const size_t stride = (n >> 1) / half;
                const size_t current_n = half << 1;
                for (size_t i = 0; i < n; i += current_n) {
                    C* __restrict group_a = &data[i];
                    C* __restrict group_b = &data[i + half];
                    for (size_t k = 0; k < half; ++k) {
                        const auto w = w_ptr[k * stride];

                        const auto a = group_a[k];
                        const auto b = w * group_b[k];

                        group_a[k] = a + b;
                        group_b[k] = a - b;
                    }
                }
            }
        }

    private:
        std::vector<C> twiddles_;
    };
}
