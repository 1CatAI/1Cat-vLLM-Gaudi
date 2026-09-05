# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.18)
project(flashinfer_gaudi_kernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_program(TPC_COMPILER tpc-clang REQUIRED)
find_program(ELF_OBJCOPY objcopy REQUIRED)
set(TPC_INCLUDE_DIR "/usr/lib/habanatools/include" CACHE PATH "TPC compiler include directory")
set(HABANA_INCLUDE_DIR "/usr/include/habanalabs" CACHE PATH "Gaudi host API include directory")

set(TPC_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gdn_packed_decode_f32_gaudi2.c")
set(TPC_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/gdn_packed_decode_f32_gaudi2.o")
set(TPC_EMBEDDED_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/gdn_packed_decode_f32_gaudi2_embedded.o")

add_custom_command(
    OUTPUT "${TPC_OBJECT}"
    COMMAND "${TPC_COMPILER}" -Wall -Werror -O2 -march=gaudi2
            -I"${TPC_INCLUDE_DIR}" "${TPC_SOURCE}" -c -o "${TPC_OBJECT}"
    DEPENDS "${TPC_SOURCE}"
    VERBATIM
)

add_custom_command(
    OUTPUT "${TPC_EMBEDDED_OBJECT}"
    COMMAND "${ELF_OBJCOPY}" -I binary -O elf64-x86-64 -B i386:x86-64
            gdn_packed_decode_f32_gaudi2.o gdn_packed_decode_f32_gaudi2_embedded.o
    DEPENDS "${TPC_OBJECT}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    VERBATIM
)

set_source_files_properties("${TPC_EMBEDDED_OBJECT}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)

add_library(
    flashinfer_gaudi_kernels
    SHARED
    host/entry_points.cpp
    host/gdn_packed_decode_f32_gaudi2.cpp
    "${TPC_EMBEDDED_OBJECT}"
)
target_include_directories(
    flashinfer_gaudi_kernels
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/host"
    "${TPC_INCLUDE_DIR}"
    "${HABANA_INCLUDE_DIR}"
)
target_compile_options(flashinfer_gaudi_kernels PRIVATE -Wall -Wextra -Werror -fno-strict-aliasing)
target_link_options(flashinfer_gaudi_kernels PRIVATE -Wl,-z,noexecstack)
