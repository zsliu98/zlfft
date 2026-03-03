cmake_minimum_required(VERSION 3.20)
project(zlfft)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(ACCURACY_TEST "Accuracy Test" ON)
option(THROUGHPUT_TEST "Throughput Test" OFF)

if (ACCURACY_TEST)
    add_executable(zlfft_benchmark benchmark/accuracy.cpp)
endif ()
if (THROUGHPUT_TEST)
    add_executable(zlfft_benchmark benchmark/throughput.cpp)

    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing Google benchmark's tests" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Suppressing Google benchmark's install" FORCE)
    set(BENCHMARK_DOWNLOAD_DEPENDENCIES OFF CACHE BOOL "Suppressing Google benchmark's dependencies downloading" FORCE)
    add_subdirectory(google/benchmark)
    target_link_libraries(zlfft_benchmark PRIVATE benchmark::benchmark)
endif ()

target_include_directories(zlfft_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/benchmark ${CMAKE_CURRENT_SOURCE_DIR}/source)

option(ENABLE_NAIVE_COOLEY_RADIX2 "Enable naive_cooley_radix2" OFF)
if (ENABLE_NAIVE_COOLEY_RADIX2)
    target_compile_definitions(zlfft_benchmark PRIVATE ENABLE_NAIVE_COOLEY_RADIX2)
endif ()

option(ENABLE_STOCKHAM_RADIX2_KERNEL24 "Enable stockham_radix2_kernel24" OFF)
if (ENABLE_STOCKHAM_RADIX2_KERNEL24)
    target_compile_definitions(zlfft_benchmark PRIVATE ENABLE_STOCKHAM_RADIX2_KERNEL24)
endif ()

option(ENABLE_NAIVE_STOCKHAM_RADIX4 "Enable naive_stockham_randix4" OFF)
if (ENABLE_NAIVE_STOCKHAM_RADIX4)
    target_compile_definitions(zlfft_benchmark PRIVATE ENABLE_NAIVE_STOCKHAM_RADIX4)
endif ()

option(ENABLE_STOCKHAM_RADIX2_KERNEL248 "Enable stockham_radix2_kernel248" OFF)
if (ENABLE_STOCKHAM_RADIX2_KERNEL248)
    target_compile_definitions(zlfft_benchmark PRIVATE ENABLE_STOCKHAM_RADIX2_KERNEL248)
endif ()

option(ENABLE_SIMD_STOCKHAM_RADIX2 "Enable simd_stockham_radix2" OFF)
if (ENABLE_SIMD_STOCKHAM_RADIX2)
    target_compile_definitions(zlfft_benchmark PRIVATE ENABLE_SIMD_STOCKHAM_RADIX2)
endif ()

option(ENABLE_SIMD_STOCKHAM_RADIX2_KERNEL1 "Enable simd_stockham_radix2_kernel1" OFF)
if (ENABLE_SIMD_STOCKHAM_RADIX2_KERNEL1)
    target_compile_definitions(zlfft_benchmark PRIVATE ENABLE_SIMD_STOCKHAM_RADIX2_KERNEL1)
endif ()

option(ENABLE_SIMD_STOCKHAM_RADIX2_KERNEL2 "Enable simd_stockham_radix2_kernel2" OFF)
if (ENABLE_SIMD_STOCKHAM_RADIX2_KERNEL2)
    target_compile_definitions(zlfft_benchmark PRIVATE ENABLE_SIMD_STOCKHAM_RADIX2_KERNEL2)
endif ()

option(ENABLE_SIMD_STOCKHAM_RADIX2_KERNEL24 "Enable simd_stockham_radix2_kernel24" OFF)
if (ENABLE_SIMD_STOCKHAM_RADIX2_KERNEL24)
    target_compile_definitions(zlfft_benchmark PRIVATE ENABLE_SIMD_STOCKHAM_RADIX2_KERNEL24)
endif ()

option(ENABLE_KFR "Enable kfr" OFF)
if (ENABLE_KFR)
    set(KFR_USE_BOOST_MATH OFF CACHE BOOL "Disable Boost Math usage in kfr" FORCE)
    set(KFR_USE_BOOST OFF CACHE BOOL "Disable Boost usage in kfr" FORCE)
    set(KFR_ENABLE_MULTIARCH OFF CACHE BOOL "Disable kfr SIMD runtime dispatch" FORCE)
    if(NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm64"))
        set(KFR_ARCH "avx2" CACHE STRING "Set SIMD set to AVX2" FORCE)
    endif()
    add_subdirectory(kfr_impl/kfr)
    target_link_libraries(zlfft_benchmark PRIVATE kfr kfr_dft)
    target_compile_definitions(zlfft_benchmark PRIVATE ENABLE_KFR)
endif ()

option(ENABLE_FFTW3 "Enable fftw3" OFF)
if (ENABLE_FFTW3)
    set(BUILD_TESTS OFF CACHE BOOL "Disable fftw3 tests" FORCE)
    set(ENABLE_FLOAT ON CACHE BOOL "Enable single precision support" FORCE)
    add_subdirectory(fftw3_impl/fftw3 EXCLUDE_FROM_ALL)
    
    target_link_libraries(zlfft_benchmark PRIVATE fftw3f)
    target_include_directories(zlfft_benchmark PRIVATE fftw3_impl/fftw3/api)
    target_compile_definitions(zlfft_benchmark PRIVATE ENABLE_FFTW3)
endif ()

set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "Disable Highway examples" FORCE)
set(HWY_ENABLE_TESTS OFF CACHE BOOL "Disable Highway tests" FORCE)
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "Disable Highway contrib" FORCE)

add_subdirectory(google/highway)
target_link_libraries(zlfft_benchmark PRIVATE hwy)

if (MSVC)
    target_compile_options(zlfft_benchmark PRIVATE $<$<CONFIG:RELEASE>:/fp:precise>)
else ()
    target_compile_options(zlfft_benchmark PRIVATE $<$<CONFIG:RELEASE>:-O3 -ffp-contract=fast -fno-signed-zeros -freciprocal-math>)
    target_compile_options(zlfft_benchmark PRIVATE $<$<CONFIG:RelWithDebInfo>:-O3 -ffp-contract=fast -fno-signed-zeros -freciprocal-math>)
endif ()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    target_compile_options(zlfft_benchmark PRIVATE "-march=armv8-a+simd")
else ()
    target_compile_options(zlfft_benchmark PRIVATE "-mavx2")
endif ()

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT error)

if (ipo_supported)
    message(STATUS "LTO/IPO is supported and enabled.")
    set_target_properties(zlfft_benchmark PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
else ()
    message(WARNING "LTO/IPO is not supported: ${error}")
endif ()
