cmake_minimum_required(VERSION 3.20)
project(zlfft)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Gather all files from the source directory
file(GLOB_RECURSE SourceFiles
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/*.hpp"
)

# 2. Define the executable with just main.cpp initially
add_executable(zlfft main.cpp)

# 3. Add the gathered source files to that same target
target_sources(zlfft PRIVATE ${SourceFiles})

if (MSVC)
    target_compile_options(zlfft PRIVATE $<$<CONFIG:RELEASE>:/fp:fast>)
    target_compile_options(zlfft PRIVATE $<$<CONFIG:RELEASE>:/Ox>)
else ()
    target_compile_options(zlfft PRIVATE $<$<CONFIG:RELEASE>:-Ofast>)
    target_compile_options(zlfft PRIVATE $<$<CONFIG:RelWithDebInfo>:-Ofast>)
endif ()

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT error)

if(ipo_supported)
    message(STATUS "LTO/IPO is supported and enabled.")
    set_target_properties(zlfft PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    message(WARNING "LTO/IPO is not supported: ${error}")
endif()
