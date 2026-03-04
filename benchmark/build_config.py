import argparse
import subprocess
import os
import sys
import platform
import json

algos = ["naive_stockham_radix2", "naive_cooley_radix2", "naive_stockham_radix4",
         "stockham_radix2_kernel24", "stockham_radix2_kernel248",

         "simd_stockham_radix2",
         "simd_stockham_radix2_kernel1",
         "simd_stockham_radix2_kernel2",
         "simd_stockham_radix2_kernel24",
         "simd_stockham_radix2_kernel248",
         "simd_stockham_radix4",

         "fftw3", "kfr"]


def build_benchmark(algorithm, benchmark_type):
    build_dir = "build_accuracy"
    os.makedirs(build_dir, exist_ok=True)

    cmake_cmd = ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"]
    if benchmark_type == "accuracy":
        cmake_cmd += ["-DACCURACY_TEST=ON", "-DTHROUGHPUT_TEST=OFF"]
    elif benchmark_type == "throughput":
        cmake_cmd += ["-DACCURACY_TEST=OFF", "-DTHROUGHPUT_TEST=ON"]

    if platform.system() == "Windows":
        cmake_cmd += ["-DCMAKE_C_COMPILER=clang-cl", "-DCMAKE_CXX_COMPILER=clang-cl"]
    if platform.system() == "Linux":
        cmake_cmd += ["-DCMAKE_C_COMPILER=clang", "-DCMAKE_CXX_COMPILER=clang++"]

    for algo in algos:
        if algo == algorithm:
            cmake_cmd.append(f"-DENABLE_{algo.upper()}=ON")
        else:
            cmake_cmd.append(f"-DENABLE_{algo.upper()}=OFF")

    print(f"Configuring: {' '.join(cmake_cmd)}")
    subprocess.run(cmake_cmd, capture_output=True, cwd=build_dir, check=True)

    build_cmd = ["cmake", "--build", ".", "--target", "zlfft_benchmark", "--config", "Release", "-j"]
    print(f"Building: {' '.join(build_cmd)}")
    subprocess.run(build_cmd, capture_output=True, cwd=build_dir, check=True)

    return os.path.join(build_dir, "zlfft_benchmark")
