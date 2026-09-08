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

set(TPC_EMBEDDED_OBJECTS)
foreach(KERNEL
        gdn_packed_decode_f32_gaudi2
        gdn_mtp_packed_f32_gaudi2
        gdn_mtp_prepared_f32_gaudi2
        dflash2_select_path_i64_f32_gaudi2
        dflash2_score_select_i64_bf16_f32_gaudi2
        silu_and_mul_bf16_gaudi2
        silu_mul_quant_bf16_gaudi2
        block_fp8_dequant_gaudi2)
    set(TPC_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/${KERNEL}.c")
    set(TPC_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/${KERNEL}.o")
    set(TPC_EMBEDDED_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/${KERNEL}_embedded.o")
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
                "${KERNEL}.o" "${KERNEL}_embedded.o"
        DEPENDS "${TPC_OBJECT}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        VERBATIM
    )
    set_source_files_properties("${TPC_EMBEDDED_OBJECT}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
    list(APPEND TPC_EMBEDDED_OBJECTS "${TPC_EMBEDDED_OBJECT}")
endforeach()

add_library(
    flashinfer_gaudi_kernels
    SHARED
    host/dflash2_score_select_i64_bf16_f32_gaudi2.cpp
    host/dflash2_select_path_i64_f32_gaudi2.cpp
    host/entry_points.cpp
    host/gdn_mtp_packed_f32_gaudi2.cpp
    host/gdn_mtp_prepared_f32_gaudi2.cpp
    host/gdn_packed_decode_f32_gaudi2.cpp
    host/silu_and_mul_bf16_gaudi2.cpp
    host/silu_mul_quant_bf16_gaudi2.cpp
    host/block_fp8_dequant_gaudi2.cpp
    ${TPC_EMBEDDED_OBJECTS}
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

option(FLASHINFER_GAUDI_BUILD_TESTS "Build CPU kernel-glue contract tests" OFF)

# Separate research database: existing FlashInfer GUIDs and libraries retain
# their current registration and qualification.
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/tp2_dynamic_quant_bf16_gaudi2.o"
    COMMAND "${TPC_COMPILER}" -Wall -Werror -O2 -march=gaudi2 -I"${TPC_INCLUDE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/tp2_dynamic_quant_bf16_gaudi2.c"
            -c -o "${CMAKE_CURRENT_BINARY_DIR}/tp2_dynamic_quant_bf16_gaudi2.o"
    DEPENDS kernels/tp2_dynamic_quant_bf16_gaudi2.c VERBATIM)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/tp2_dynamic_quant_bf16_gaudi2_embedded.o"
    COMMAND "${ELF_OBJCOPY}" -I binary -O elf64-x86-64 -B i386:x86-64
            tp2_dynamic_quant_bf16_gaudi2.o tp2_dynamic_quant_bf16_gaudi2_embedded.o
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/tp2_dynamic_quant_bf16_gaudi2.o"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}" VERBATIM)
set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/tp2_dynamic_quant_bf16_gaudi2_embedded.o"
    PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
add_library(tp2_dynamic_quant_kernels SHARED host/tp2_quant_entry.cpp host/tp2_dynamic_quant_bf16_gaudi2.cpp
    "${CMAKE_CURRENT_BINARY_DIR}/tp2_dynamic_quant_bf16_gaudi2_embedded.o")
target_include_directories(tp2_dynamic_quant_kernels PRIVATE host "${TPC_INCLUDE_DIR}" "${HABANA_INCLUDE_DIR}")
target_compile_options(tp2_dynamic_quant_kernels PRIVATE -Wall -Wextra -Werror -fno-strict-aliasing)
target_link_options(tp2_dynamic_quant_kernels PRIVATE -Wl,-z,noexecstack)
if(FLASHINFER_GAUDI_BUILD_TESTS)
    enable_testing()
    add_executable(flashinfer_gaudi_host_test host/test_kernel_glue.cpp)
    target_include_directories(flashinfer_gaudi_host_test PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/host" "${HABANA_INCLUDE_DIR}" "${TPC_INCLUDE_DIR}")
    target_compile_options(flashinfer_gaudi_host_test PRIVATE -UNDEBUG)
    target_link_libraries(flashinfer_gaudi_host_test PRIVATE flashinfer_gaudi_kernels)
    add_test(NAME flashinfer_gaudi_kernel_glue COMMAND flashinfer_gaudi_host_test)
endif()
