#include "benchmark_include.hpp"

int main(const int, char** argv) {
    const int order = std::stoi(argv[1]);
    const size_t num_repeats = std::stoi(argv[2]);
    const size_t n = static_cast<size_t>(1) << order;

    std::vector<C> in(n);
    generate_random_data(in);
    std::vector<C> out(n);

    FFTClass fft(order);

    // warm up
    for (size_t i = 0; i < 50; ++i) {
        fft.forward(in, out);
        fft.forward(out, in);
    }

    const auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < (num_repeats >> 1); ++i) {
        fft.forward(in, out);
        fft.forward(out, in);
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const auto total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    const double avg_time_us = (total_time_ms * 1000.0) / num_repeats;
    std::cout << std::scientific << avg_time_us << std::endl;
    return 0;
}
