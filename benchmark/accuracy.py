import argparse
import subprocess
import os
import sys
import platform

algos = ["naive_stockham_radix2", "naive_cooley_radix2", "naive_stockham_radix4",
         "stockham_radix2_kernel4", "stockham_radix2_kernel8",
         "fftw3", "kfr"]


def build_benchmark(algorithm):
    build_dir = "build_accuracy"
    os.makedirs(build_dir, exist_ok=True)

    cmake_cmd = ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release", "-DACCURACY_TEST=ON", "-DTHROUGHPUT_TEST=OFF"]

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


def run_benchmark(exe_path, n0, n1, algorithm):
    print(f"Running accuracy benchmark for {algorithm} from order {n0} to {n1}...")
    print(f"{'Order':<10} {'MSE':<18}")
    print("-" * 28)

    mses = []
    for n in range(n0, n1 + 1):
        cmd = [exe_path, str(n)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        mse = float(result.stdout)
        mses.append(mse)
        print(f"{n:<10} {mse:<18.8e}")

    print()
    print(mses)


def main():
    parser = argparse.ArgumentParser(description="Accuracy Benchmark for FFT")
    parser.add_argument("n0", type=int, help="Start FFT order (size 2^n)")
    parser.add_argument("n1", type=int, help="End FFT order (size 2^n)")
    parser.add_argument("algorithm", type=str, help="Algorithm to test (e.g., kfr)")

    args = parser.parse_args()

    try:
        exe_path = build_benchmark(args.algorithm)
        run_benchmark(exe_path, args.n0, args.n1, args.algorithm)
    except subprocess.CalledProcessError as e:
        print(f"Error executing benchmark: {e}")
        print(f"Stdout:\n{e.stdout}")
        print(f"Stderr:\n{e.stderr}")
        sys.exit(1)


if __name__ == "__main__":
    main()
