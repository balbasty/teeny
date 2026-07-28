# teeny — header-only C++17 tensor library (host + CUDA), built on CCCL's
# cuda::std::mdspan. This script exposes teeny as an INTERFACE target so a
# consumer just links it and inherits the include dirs, the C++17 requirement,
# and the CCCL dependency — nothing to compile.
#
#   Consume in-tree:   add_subdirectory(teeny)      then  target_link_libraries(app PRIVATE teeny::teeny)
#   Consume installed: find_package(teeny CONFIG)   then  target_link_libraries(app PRIVATE teeny::teeny)
#
# The Makefile remains the reference build for the test suite; this file is for
# downstream projects that organise with CMake (issue #111).

cmake_minimum_required(VERSION 3.14)   # INTERFACE targets + FetchContent + install(EXPORT)

project(teeny
    VERSION 0.1.0
    DESCRIPTION "A tiny static tensor library (host + CUDA) on cuda::std::mdspan"
    HOMEPAGE_URL "https://github.com/balbasty/teeny"
    LANGUAGES CXX)

# Are we the top-level project, or pulled in via add_subdirectory/FetchContent?
# (Gate tests/examples and install on this, so we don't impose them on parents.)
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(TEENY_IS_TOP_LEVEL ON)
else()
    set(TEENY_IS_TOP_LEVEL OFF)
endif()

option(TEENY_BUILD_TESTS    "Build the teeny test suite (CTest)" ${TEENY_IS_TOP_LEVEL})
option(TEENY_BUILD_EXAMPLES "Build the teeny examples"           OFF)

# ---------------------------------------------------------------------------
# The library target
# ---------------------------------------------------------------------------
add_library(teeny INTERFACE)
add_library(teeny::teeny ALIAS teeny)

target_compile_features(teeny INTERFACE cxx_std_17)   # the floor CCCL enforces

# SCRATCH (investigation only, see #316): try MSVC's conformant parser to see
# whether it collapses several MSVC-only defects (#294/#296/#311/#315) at once.
target_compile_options(teeny INTERFACE $<$<CXX_COMPILER_ID:MSVC>:/permissive->)

target_include_directories(teeny INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>")

# ---------------------------------------------------------------------------
# CCCL (libcudacxx — cuda::std::mdspan). Resolution order:
#   1. TEENY_CCCL_INCLUDE — an explicit include dir you point at a specific CCCL
#      (the knob for a per-CUDA choice, e.g. a newer 3.x for CUDA 13; mirrors the
#      Makefile's CCCL_INC). See docs/cuda-compat.md.
#   2. an installed CCCL via find_package(CCCL).
#   3. the vendored submodule (pinned to CCCL v2.8.2 — supports CUDA 11.1+).
# `find_package(teeny)` re-resolves CCCL via find_dependency (teeny-config.cmake.in),
# so an installed teeny needs an installed CCCL.
# ---------------------------------------------------------------------------
set(TEENY_CCCL_INCLUDE "" CACHE PATH
    "Explicit CCCL (libcudacxx) include dir; empty = installed CCCL, else vendored submodule")
if (TEENY_CCCL_INCLUDE)
    message(STATUS "teeny: using CCCL from TEENY_CCCL_INCLUDE=${TEENY_CCCL_INCLUDE}")
    target_include_directories(teeny INTERFACE "$<BUILD_INTERFACE:${TEENY_CCCL_INCLUDE}>")
else()
    find_package(CCCL CONFIG QUIET)
    if (CCCL_FOUND)
        target_link_libraries(teeny INTERFACE CCCL::CCCL)
    else()
        set(_teeny_cccl "${CMAKE_CURRENT_SOURCE_DIR}/external/cccl/libcudacxx/include")
        if (EXISTS "${_teeny_cccl}/cuda/std/mdspan")
            message(STATUS "teeny: using vendored CCCL at external/cccl (in-tree build)")
            # BUILD_INTERFACE only: the vendored path is not valid post-install.
            target_include_directories(teeny INTERFACE "$<BUILD_INTERFACE:${_teeny_cccl}>")
        else()
            message(FATAL_ERROR
                "teeny requires CCCL (cuda::std::mdspan) but none was found.\n"
                "  - set -DTEENY_CCCL_INCLUDE=/path/to/cccl/include, or\n"
                "  - install CCCL and it will be picked up via find_package(CCCL), or\n"
                "  - initialise the vendored submodule:\n"
                "      git submodule update --init --depth 1 external/cccl")
        endif()
    endif()
endif()

# ---------------------------------------------------------------------------
# Tests (each tests/test_*.cpp is a standalone executable → CTest)
# ---------------------------------------------------------------------------
if (TEENY_BUILD_TESTS)
    enable_testing()
    file(GLOB _teeny_tests CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.cpp")
    foreach (_src IN LISTS _teeny_tests)
        get_filename_component(_name "${_src}" NAME_WE)
        add_executable(${_name} "${_src}")
        target_link_libraries(${_name} PRIVATE teeny::teeny)
        # test_cuda compiles against the malloc-backed fake CUDA runtime.
        if (_name STREQUAL "test_cuda")
            target_include_directories(${_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests/fakecuda")
        endif()
        add_test(NAME ${_name} COMMAND ${_name})
    endforeach()
endif()

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
if (TEENY_BUILD_EXAMPLES)
    file(GLOB _teeny_examples CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/examples/ex_*.cpp")
    foreach (_src IN LISTS _teeny_examples)
        get_filename_component(_name "${_src}" NAME_WE)
        add_executable(${_name} "${_src}")
        target_link_libraries(${_name} PRIVATE teeny::teeny)
    endforeach()
endif()

# ---------------------------------------------------------------------------
# Install + package config (so downstream `find_package(teeny CONFIG)` works)
# ---------------------------------------------------------------------------
if (TEENY_IS_TOP_LEVEL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/teeny"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

    install(TARGETS teeny EXPORT teeny-targets)
    install(EXPORT teeny-targets
            FILE teeny-targets.cmake
            NAMESPACE teeny::
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/teeny")

    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/teeny-config.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/teeny-config.cmake"
        INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/teeny")

    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/teeny-config-version.cmake"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY SameMajorVersion
        ARCH_INDEPENDENT)   # header-only: no ABI, so any arch matches

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/teeny-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/teeny-config-version.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/teeny")
endif()
