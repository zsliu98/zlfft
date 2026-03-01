import argparse
import subprocess
import os
import sys
import platform

algos = ["kfr"]


def build_benchmark(algorithm):
    build_dir = "build_throughput"
    os.makedirs(build_dir, exist_ok=True)

    cmake_cmd = ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release", "-DACCURACY_TEST=OFF", "-DTHROUGHPUT_TEST=ON"]

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
    subprocess.run(cmake_cmd, cwd=build_dir, check=True)

    build_cmd = ["cmake", "--build", ".", "--target", "zlfft_benchmark", "--config", "Release", "-j"]
    print(f"Building: {' '.join(build_cmd)}")
    subprocess.run(build_cmd, cwd=build_dir, check=True)

    return os.path.join(build_dir, "zlfft_benchmark")


def run_benchmark(exe_path, n0, n1, algorithm, num_repeats):
    print(f"Running throughput benchmark for {algorithm} from order {n0} to {n1} ({num_repeats} repeats)...")
    for n in range(n0, n1 + 1):
        cmd = [exe_path, str(n), str(int(num_repeats * n1 / n))]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        avg_time = float(result.stdout)
        throughput = 5 * (2 ** n) * n / avg_time
        print("order={}, avg_time={}, throughput={}".format(n, avg_time, throughput))


def main():
    parser = argparse.ArgumentParser(description="Throughput Benchmark for FFT")
    parser.add_argument("n0", type=int, help="Start FFT order (size 2^n)")
    parser.add_argument("n1", type=int, help="End FFT order (size 2^n)")
    parser.add_argument("algorithm", type=str, help="Algorithm to test (e.g., naive_stockham_dit_radix2, kfr)")
    parser.add_argument("num_repeats", type=int, help="Number of repetitions per size")

    args = parser.parse_args()

    try:
        exe_path = build_benchmark(args.algorithm)
        run_benchmark(exe_path, args.n0, args.n1, args.algorithm, args.num_repeats)
    except subprocess.CalledProcessError as e:
        print(f"Error executing benchmark: {e}")
        print(f"Stdout:\n{e.stdout}")
        print(f"Stderr:\n{e.stderr}")
        sys.exit(1)


if __name__ == "__main__":
    main()
