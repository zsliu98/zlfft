import argparse
import subprocess
import os
import sys
import platform
import json
import time

from build_config import build_benchmark


def run_benchmark(exe_path, n0, n1, algorithm):
    print(f"Running throughput benchmark for {algorithm} from order {n0} to {n1}...")
    cmd = [exe_path, str(n0), str(n1), "--benchmark_format=json"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Benchmark failed for {algorithm}.\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
        return None

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON output for {algorithm}.")
        return None

    throughputs = []
    for bench in data.get("benchmarks", []):
        name = bench["name"]
        try:
            n = int(name.split('/')[-1])
        except ValueError:
            continue

        cpu_time_us = bench["cpu_time"] * 0.5

        ops = 5 * (2 ** n) * n
        throughput = ops / cpu_time_us
        throughputs.append(throughput)

    return throughputs


def main():
    parser = argparse.ArgumentParser(description="Batch Throughput Benchmark for FFT Algorithms")
    parser.add_argument("n0", type=int, help="Start FFT order (size 2^n)")
    parser.add_argument("n1", type=int, help="End FFT order (size 2^n)")
    parser.add_argument("--avx2", action="store_true", help="Enable AVX2 architecture")

    args = parser.parse_args()

    if platform.system() == "Darwin":
        algorithms = [
            "fftw3", "fftw3_estimate", "kfr", "vdsp", "pffft", 
            "simd_low_order_opt1", "simd_low_order_opt2", 
            "simd_low_order_aosoa1", "simd_low_order_aosoa2"
        ]
    else:
        algorithms = [
            "fftw3", "fftw3_estimate", "kfr", "ipp", "pffft", 
            "simd_low_order_opt1", "simd_low_order_opt2", 
            "simd_low_order_aosoa1", "simd_low_order_aosoa2"
        ]

    results = {}

    for i, algo in enumerate(algorithms):
        if i > 0:
            print("\nSleeping 10 seconds to let the CPU cool down...")
            time.sleep(10)
            
        print(f"\n[{i+1}/{len(algorithms)}] Building {algo}...")
        
        try:
            exe_path = build_benchmark(algo, "throughput", use_avx2=args.avx2)
            time.sleep(10)
            throughput_data = run_benchmark(exe_path, args.n0, args.n1, algo)
            
            if throughput_data is not None:
                results[algo] = throughput_data
            else:
                print(f"Skipping {algo} due to execution failure.")
                
        except Exception as e:
            print(f"Error building or running benchmark for {algo}: {e}")
            continue

    print("\n" + "="*50)
    print("FINAL BENCHMARK RESULTS")
    print("="*50)
    print(algo)
    print(str(results).replace("],", "],\n"))


if __name__ == "__main__":
    main()