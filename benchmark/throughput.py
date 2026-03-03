import argparse
import subprocess
import os
import sys
import platform
import json

algos = ["naive_stockham_radix2", "naive_cooley_radix2", "naive_stockham_radix4",
         "stockham_radix2_kernel4", "stockham_radix2_kernel8",
         "simd_stockham_radix2",
         "fftw3", "kfr"]


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
    subprocess.run(cmake_cmd, capture_output=True, cwd=build_dir, check=True)

    build_cmd = ["cmake", "--build", ".", "--target", "zlfft_benchmark", "--config", "Release", "-j"]
    print(f"Building: {' '.join(build_cmd)}")
    subprocess.run(build_cmd, capture_output=True, cwd=build_dir, check=True)

    return os.path.join(build_dir, "zlfft_benchmark")


def run_benchmark(exe_path, n0, n1, algorithm):
    print(f"Running throughput benchmark for {algorithm} from order {n0} to {n1}...")
    cmd = [exe_path, str(n0), str(n1), "--benchmark_format=json"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Benchmark failed.\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
        return

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        print("Failed to parse JSON output.")
        return

    print(f"{'Order':<10} {'Time (us)':<15} {'Throughput (MFLOPS)':<20}")
    print("-" * 45)

    cpu_times = []
    throughputs = []
    for bench in data.get("benchmarks", []):
        name = bench["name"]
        try:
            n = int(name.split('/')[-1])
        except ValueError:
            continue

        cpu_time_us = bench["cpu_time"] * 0.5
        cpu_times.append(cpu_time_us)

        ops = 5 * (2 ** n) * n
        throughput = ops / cpu_time_us
        throughputs.append(throughput)

        print(f"{n:<10} {cpu_time_us:<15.4f} {throughput:<20.4f}")

    print()
    print(cpu_times)
    print(throughputs)


def main():
    parser = argparse.ArgumentParser(description="Throughput Benchmark for FFT")
    parser.add_argument("n0", type=int, help="Start FFT order (size 2^n)")
    parser.add_argument("n1", type=int, help="End FFT order (size 2^n)")
    parser.add_argument("algorithm", type=str, help="Algorithm to test (e.g., naive_stockham_radix2, kfr)")

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
