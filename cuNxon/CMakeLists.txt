# =============================================================================
#  cuNxon  -  CMake build
#  Produces:
#     libcunxon_static.a / libcunxon_static.lib       (static lib)
#     libcunxon.so / cunxon.dll                       (shared lib)
#     example_4sphere                                 (demo executable)
#
#  Tested with:  CMake >= 3.18,  nvcc >= 11.0,  GCC >= 9 / MSVC 19.30+
#  Default arch: sm_70  (override with -DCUNXON_CUDA_ARCH=86 etc.)
# =============================================================================
cmake_minimum_required(VERSION 3.18)

project(cuNxon
        VERSION 0.1.0
        DESCRIPTION "CUDA library for Multi-Neuraxon 2.0 networks"
        LANGUAGES C CXX CUDA)

# ---- options ----------------------------------------------------------------
option(CUNXON_BUILD_SHARED   "Build shared library libcunxon.so"  ON)
option(CUNXON_BUILD_STATIC   "Build static library libcunxon.a"   ON)
option(CUNXON_BUILD_EXAMPLES "Build example programs"             ON)
option(CUNXON_BUILD_TESTS    "Build the test_cunxon test suite"   ON)

# Comma-separated list of CC numbers (e.g. "70;75;80;86;89;90").
set(CUNXON_CUDA_ARCH "70" CACHE STRING
    "Semicolon-separated list of CUDA compute capabilities (e.g. 70;80;86)")

# ---- standards / flags ------------------------------------------------------
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
set(CMAKE_CUDA_ARCHITECTURES ${CUNXON_CUDA_ARCH})

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Position-independent code for shared lib + clean compiler diagnostics
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Pass through extra flags only in Release
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
    if (NOT MSVC)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-O3>)
    endif()
endif()

# Quieter nvcc
add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=20012>   # constexpr in __device__
)

# ---- sources ----------------------------------------------------------------
set(CUNXON_SOURCES
    src/cuNxon.cu
    src/cuNxon_kernels.cu
    src/cuNxon_plasticity.cu
    src/cuNxon_intersphere.cu)

set(CUNXON_HEADERS
    include/cuNxon.h
    include/cuNxon_types.h)

# ---- find CUDA toolkit (for cuRAND) -----------------------------------------
find_package(CUDAToolkit REQUIRED)

# ---- static library ---------------------------------------------------------
if (CUNXON_BUILD_STATIC)
    add_library(cunxon_static STATIC ${CUNXON_SOURCES} ${CUNXON_HEADERS})
    target_include_directories(cunxon_static
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(cunxon_static PUBLIC CUDA::curand)
    # Static build: no __declspec on functions, in both library and consumer.
    target_compile_definitions(cunxon_static PUBLIC CUNXON_STATIC)
    set_target_properties(cunxon_static PROPERTIES
        OUTPUT_NAME cunxon_static
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON)
endif()

# ---- shared library ---------------------------------------------------------
if (CUNXON_BUILD_SHARED)
    add_library(cunxon SHARED ${CUNXON_SOURCES} ${CUNXON_HEADERS})
    target_include_directories(cunxon
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(cunxon PUBLIC CUDA::curand)
    # PRIVATE: only the DLL itself sees CUNXON_BUILDING_DLL (=> dllexport).
    # Consumers without it (the default) will see dllimport on Windows.
    target_compile_definitions(cunxon PRIVATE CUNXON_BUILDING_DLL)
    set_target_properties(cunxon PROPERTIES
        VERSION   ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        CUDA_VISIBILITY_PRESET hidden)
endif()

# Convenience alias: prefer shared, fall back to static, for downstream targets
if (CUNXON_BUILD_SHARED)
    add_library(cuNxon::cuNxon ALIAS cunxon)
elseif (CUNXON_BUILD_STATIC)
    add_library(cuNxon::cuNxon ALIAS cunxon_static)
endif()

# ---- examples ---------------------------------------------------------------
if (CUNXON_BUILD_EXAMPLES)
    add_executable(example_4sphere examples/example_4sphere.cu)
    target_link_libraries(example_4sphere PRIVATE cuNxon::cuNxon)
    set_target_properties(example_4sphere PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

    add_executable(example_aigarth examples/example_aigarth.cu)
    target_link_libraries(example_aigarth PRIVATE cuNxon::cuNxon)
    set_target_properties(example_aigarth PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
endif()

# ---- tests ------------------------------------------------------------------
if (CUNXON_BUILD_TESTS)
    enable_testing()
    add_executable(test_cunxon tests/test_cunxon.cu)
    target_link_libraries(test_cunxon PRIVATE cuNxon::cuNxon)
    set_target_properties(test_cunxon PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
    add_test(NAME cunxon_smoke COMMAND test_cunxon)
endif()

# ---- install rules ----------------------------------------------------------
include(GNUInstallDirs)

set(CUNXON_INSTALL_TARGETS "")
if (CUNXON_BUILD_STATIC)
    list(APPEND CUNXON_INSTALL_TARGETS cunxon_static)
endif()
if (CUNXON_BUILD_SHARED)
    list(APPEND CUNXON_INSTALL_TARGETS cunxon)
endif()

if (CUNXON_INSTALL_TARGETS)
    install(TARGETS ${CUNXON_INSTALL_TARGETS}
            EXPORT  cuNxonTargets
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

install(FILES ${CUNXON_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT cuNxonTargets
        FILE        cuNxonTargets.cmake
        NAMESPACE   cuNxon::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuNxon)

# ---- status summary ---------------------------------------------------------
message(STATUS "--------------------------------------------------")
message(STATUS "  cuNxon ${PROJECT_VERSION}")
message(STATUS "    CUDA arch     : ${CUNXON_CUDA_ARCH}")
message(STATUS "    static lib    : ${CUNXON_BUILD_STATIC}")
message(STATUS "    shared lib    : ${CUNXON_BUILD_SHARED}")
message(STATUS "    examples      : ${CUNXON_BUILD_EXAMPLES}")
message(STATUS "    tests         : ${CUNXON_BUILD_TESTS}")
message(STATUS "    build type    : ${CMAKE_BUILD_TYPE}")
message(STATUS "    CXX compiler  : ${CMAKE_CXX_COMPILER}")
message(STATUS "    CUDA compiler : ${CMAKE_CUDA_COMPILER}")
message(STATUS "--------------------------------------------------")
