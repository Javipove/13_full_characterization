cmake_minimum_required(VERSION 3.24)

# Use clang-18 as compiler
find_program(CLANG_C_COMPILER clang-18)
find_program(CLANG_CXX_COMPILER clang++-18)
if(CLANG_C_COMPILER AND CLANG_CXX_COMPILER)
    set(CMAKE_C_COMPILER "${CLANG_C_COMPILER}" CACHE STRING "C compiler" FORCE)
    set(CMAKE_CXX_COMPILER "${CLANG_CXX_COMPILER}" CACHE STRING "C++ compiler" FORCE)
else()
    message(WARNING "clang-18 not found, falling back to default compiler")
endif()

project(my_tt_out_of_tree_app LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

# Match tt-metal's build type and optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DDEBUG -fno-omit-frame-pointer")

# Match tt-metal's compile flags for equivalent binary output.
# Excludes -Wall/-Werror (warnings-as-errors) intentionally.
add_compile_options(
    # Required when linking with libstdc++ with clang (aligned allocator support)
    $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-fsized-deallocation>
    # x86-64-v3 baseline (AVX2, BMI2, FMA, etc.) — matches tt-metal target arch
    "$<$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},x86_64>:-march=x86-64-v3>"
    -fPIC
    -fvisibility-inlines-hidden
    # Clang-specific warning suppressions (matching tt-metal status quo)
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-deprecated-declarations>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-c++11-narrowing>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-deprecated-volatile>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-logical-op-parentheses>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-reorder-ctor>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-unused-lambda-capture>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-unused-private-field>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-unknown-pragmas>"
    -Wno-unused-but-set-variable
)

# Match tt-metal's linker flags
add_link_options($<$<CXX_COMPILER_ID:Clang>:-Wl,--compress-debug-sections=zstd>)

# Use mold linker if available (same as tt-metal default)
if(NOT DEFINED CMAKE_LINKER_TYPE)
    find_program(MOLD ld.mold)
    if(MOLD)
        message(STATUS "Using mold linker: ${MOLD}")
        set(CMAKE_LINKER_TYPE MOLD)
    endif()
endif()

if(NOT DEFINED ENV{TT_METAL_HOME})
    message(FATAL_ERROR "Please export TT_METAL_HOME")
endif()

set(TT_METAL_HOME "$ENV{TT_METAL_HOME}")
set(TT_BUILD_DIR "${TT_METAL_HOME}/build_tracy_reldeb")

# Point cmake to all package configs
list(PREPEND CMAKE_PREFIX_PATH
    "${TT_BUILD_DIR}/lib/cmake"
    "${TT_BUILD_DIR}/share/cmake"
)

# Find packages — all transitive deps resolved automatically
if(NOT TARGET TT::Metalium)
    find_package(tt-metalium CONFIG REQUIRED)
endif()
if(NOT TARGET TTNN::TTNN)
    find_package(tt-nn CONFIG REQUIRED)
endif()

add_executable(test_full_charac test_full_charac.cpp)

# TTNN::TTNN maps to _ttnncpp.so and brings TT::Metalium transitively,
# which in turn brings umd::device, fmt, Tracy, tt_stl, nlohmann_json, etc.
target_link_libraries(test_full_charac PRIVATE
    TTNN::TTNN
    TT::Metalium
)

# Source-tree include paths for internal/private headers NOT part of the
# exported public API. Needed by util.hpp and its transitive include chain.
# Matches the in-tree perf_microbenchmark CMakeLists.txt pattern.
target_include_directories(test_full_charac BEFORE PRIVATE
    # Internal tt-metal source headers (for util.hpp transitive chain)
    ${TT_METAL_HOME}                          # "tt_metal/impl/dispatch/device_command.hpp"
    ${TT_METAL_HOME}/tt_metal/                # "impl/context/metal_context.hpp", "impl/program/..."
    ${TT_BUILD_DIR}/include/tt-metalium        # bare includes like "core_coord.hpp" in internal headers
    ${TT_METAL_HOME}/tt_metal/llrt/           # "llrt.hpp"
    ${TT_METAL_HOME}/tt_metal/common/         # "env_lib.hpp"
    ${TT_METAL_HOME}/tt_stl/                  # non-exported tt_stl headers (enum.hpp, etc.)
    # TTNN source headers (tilize.hpp not in exported FILE_SET api)
    ${TT_METAL_HOME}/ttnn/cpp
    ${TT_METAL_HOME}/ttnn/cpp/ttnn/deprecated
    # Test utilities (test_common.hpp, deprecated/tensor.hpp, util.hpp)
    ${TT_METAL_HOME}/tests
    ${TT_METAL_HOME}/tests/tt_metal/test_utils
)

set_target_properties(test_full_charac PROPERTIES
    BUILD_RPATH "${TT_BUILD_DIR}/lib"
    INSTALL_RPATH "${TT_BUILD_DIR}/lib"
)
