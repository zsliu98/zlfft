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

    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing benchmark's tests" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Suppressing benchmark's install" FORCE)
    set(BENCHMARK_DOWNLOAD_DEPENDENCIES OFF CACHE BOOL "Do not download dependencies automatically" FORCE)
    add_subdirectory(google/benchmark)
    target_link_libraries(zlfft_benchmark PRIVATE benchmark::benchmark)
endif ()

target_include_directories(zlfft_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/benchmark ${CMAKE_CURRENT_SOURCE_DIR}/source)

option(ENABLE_KFR "Enable KFR" OFF)
if (ENABLE_KFR)
    set(KFR_USE_BOOST_MATH OFF CACHE BOOL "Disable Boost Math usage in KFR" FORCE)
    set(KFR_USE_BOOST OFF CACHE BOOL "Disable Boost usage in KFR" FORCE)
    add_subdirectory(kfr_impl/kfr)
    target_link_libraries(zlfft_benchmark PRIVATE kfr kfr_dft)
    target_compile_definitions(zlfft_benchmark PRIVATE ENABLE_KFR)
endif ()

if (MSVC)
    target_compile_options(zlfft_benchmark PRIVATE $<$<CONFIG:RELEASE>:/fp:precise>)
else ()
    target_compile_options(zlfft_benchmark PRIVATE $<$<CONFIG:RELEASE>:-O3 -ffp-contract=fast -fno-signed-zeros -freciprocal-math>)
    target_compile_options(zlfft_benchmark PRIVATE $<$<CONFIG:RelWithDebInfo>:-O3 -ffp-contract=fast -fno-signed-zeros -freciprocal-math>)
endif ()

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT error)

if (ipo_supported)
    message(STATUS "LTO/IPO is supported and enabled.")
    set_target_properties(zlfft_benchmark PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
else ()
    message(WARNING "LTO/IPO is not supported: ${error}")
endif ()
