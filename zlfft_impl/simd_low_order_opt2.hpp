#pragma once

#include "zlfft_common.hpp"

namespace zlfft {
    namespace hn = hwy::HWY_NAMESPACE;

    template <typename F>
    class SIMDLowOrderOPT2 {
        using C = std::complex<F>;

    private:
        std::vector<common::StageType> stages_;

    public:
        explicit SIMDLowOrderOPT2(const size_t order) :
            order_(order) {
            if (order < 4) {
                return;
            }
            if (order == 4) {
                stages_ = {common::StageType::kRadix4FirstPass, common::StageType::kRadix4LastPass};
            } else if (order == 5) {
                stages_ = {common::StageType::kRadix8FirstPass, common::StageType::kRadix4LastPass};
            } else if (order == 6) {
                stages_ = {common::StageType::kRadix4FirstPass, common::StageType::kRadix4Width4, common::StageType::kRadix4LastPass};
            } else {
                const auto mod_result = order % 3;
                stages_.emplace_back(common::StageType::kRadix8FirstPass);
                if (mod_result == 1) {
                    for (size_t i = 3; i < order - 4; i += 3) {
                        stages_.emplace_back(common::StageType::kRadix8);
                    }
                    stages_.emplace_back(common::StageType::kRadix4);
                } else if (mod_result == 2) {
                    for (size_t i = 3; i < order - 2; i += 3) {
                        stages_.emplace_back(common::StageType::kRadix8);
                    }
                } else {
                    for (size_t i = 3; i < order - 6; i += 3) {
                        stages_.emplace_back(common::StageType::kRadix8);
                    }
                    stages_.emplace_back(common::StageType::kRadix4);
                    stages_.emplace_back(common::StageType::kRadix4);
                }
                stages_.emplace_back(common::StageType::kRadix4LastPass);
            }

            size_t num_twiddles = 0;
            {
                size_t width = (stages_[0] == common::StageType::kRadix4FirstPass) ? 4 : 8;
                for (const auto stage : stages_) {
                    switch (stage) {
                    case common::StageType::kRadix4Width4:
                    case common::StageType::kRadix4:
                    case common::StageType::kRadix4LastPass: {
                        num_twiddles += 3 * width;
                        width = width << 2;
                        break;
                    }
                    case common::StageType::kRadix8: {
                        num_twiddles += 7 * width;
                        width = width << 3;
                        break;
                    }
                    }
                }
            }

            twiddles_r_ = hwy::AllocateAligned<F>(num_twiddles);
            twiddles_i_ = hwy::AllocateAligned<F>(num_twiddles);

            {
                size_t offset = 0;
                size_t width = (stages_[0] == common::StageType::kRadix4FirstPass) ? 4 : 8;
                for (const auto stage : stages_) {
                    switch (stage) {
                    case common::StageType::kRadix4Width4:
                    case common::StageType::kRadix4:
                    case common::StageType::kRadix4LastPass: {
                        const double angle_step = -2.0 * std::numbers::pi / static_cast<double>(width << 2);
                        for (int mul = 1; mul < 4; ++mul) {
                            const auto step = angle_step * static_cast<double>(mul);
                            for (size_t k = 0; k < width; ++k, ++offset) {
                                const double angle = static_cast<double>(k) * step;
                                twiddles_r_[offset] = static_cast<F>(std::cos(angle));
                                twiddles_i_[offset] = static_cast<F>(std::sin(angle));
                            }
                        }
                        width = width << 2;
                        break;
                    }
                    case common::StageType::kRadix8: {
                        const double angle_step = -2.0 * std::numbers::pi / static_cast<double>(width << 3);
                        for (int mul = 1; mul < 8; ++mul) {
                            const auto step = angle_step * static_cast<double>(mul);
                            for (size_t k = 0; k < width; ++k, ++offset) {
                                const double angle = static_cast<double>(k) * step;
                                twiddles_r_[offset] = static_cast<F>(std::cos(angle));
                                twiddles_i_[offset] = static_cast<F>(std::sin(angle));
                            }
                        }
                        width = width << 3;
                        break;
                    }
                    }
                }
            }

            const auto n = static_cast<size_t>(1) << order;
            const auto pad = (64 / sizeof(F)) + 16;
            stride_ = n + pad;
            workspace_ = hwy::AllocateAligned<F>(4 * stride_);
        }

        void forward(std::span<C> in_buffer, std::span<C> out_buffer) {
            switch (order_) {
            case 0:
                common::callback_order_0(in_buffer.data(), out_buffer.data());
                return;
            case 1:
                common::callback_order_1(in_buffer.data(), out_buffer.data());
                return;
            case 2:
                common::callback_order_2(in_buffer.data(), out_buffer.data());
                return;
            case 3:
                common::callback_order_3(in_buffer.data(), out_buffer.data());
                return;
            case 4:
                common::callback_order_4(in_buffer.data(), out_buffer.data(),
                                         twiddles_r_.get(), twiddles_i_.get());
                return;
            case 5:
                common::callback_order_5(in_buffer.data(), out_buffer.data(),
                                         twiddles_r_.get(), twiddles_i_.get());
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

            if (stages_[0] == common::StageType::kRadix4FirstPass) {
                common::radix4_first_pass_fused(in_buffer.data(), out_r, out_i, n);
            } else {
                common::radix8_first_pass_fused(in_buffer.data(), out_r, out_i, n);
            }
            std::swap(in_r, out_r);
            std::swap(in_i, out_i);
            size_t width = (stages_[0] == common::StageType::kRadix4FirstPass) ? 4 : 8;
            for (size_t i = 1; i < stages_.size() - 1; ++i) {
                const auto stage = stages_[i];
                switch (stage) {
                case common::StageType::kRadix4Width4: {
                    common::radix4_width4(in_r, in_i, out_r, out_i, n, w_r_ptr, w_i_ptr);
                    w_r_ptr += 12;
                    w_i_ptr += 12;
                    width = width << 2;
                    break;
                }
                case common::StageType::kRadix4: {
                    common::radix4(in_r, in_i, out_r, out_i, n, width, w_r_ptr, w_i_ptr);
                    w_r_ptr += 3 * width;
                    w_i_ptr += 3 * width;
                    width = width << 2;
                    break;
                }
                case common::StageType::kRadix8: {
                    common::radix8(in_r, in_i, out_r, out_i, n, width, w_r_ptr, w_i_ptr);
                    w_r_ptr += 7 * width;
                    w_i_ptr += 7 * width;
                    width = width << 3;
                    break;
                }
                }
                std::swap(in_r, out_r);
                std::swap(in_i, out_i);
            }
            common::radix4_last_pass_fused(in_r, in_i, out_buffer.data(), n, width, w_r_ptr, w_i_ptr);
        }

    private:
        size_t order_;
        size_t stride_;
        hwy::AlignedFreeUniquePtr<F[]> twiddles_r_;
        hwy::AlignedFreeUniquePtr<F[]> twiddles_i_;
        hwy::AlignedFreeUniquePtr<F[]> workspace_;
    };
}
