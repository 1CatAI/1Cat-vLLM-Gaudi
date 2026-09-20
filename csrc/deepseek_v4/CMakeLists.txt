# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.18)
project(deepseek_v4_gaudi2 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_program(TPC_COMPILER tpc-clang REQUIRED)
find_program(ELF_OBJCOPY objcopy REQUIRED)
set(TPC_INCLUDE_DIR "/usr/lib/habanatools/include" CACHE PATH "TPC headers")
set(HABANA_INCLUDE_DIR "/usr/include/habanalabs" CACHE PATH "Gaudi host API headers")
file(GLOB KERNEL_SOURCES CONFIGURE_DEPENDS "kernels/*.c")
file(GLOB KERNEL_HEADERS CONFIGURE_DEPENDS "include/*.h")
set(EMBEDDED_OBJECTS)
foreach(SOURCE IN LISTS KERNEL_SOURCES)
    get_filename_component(NAME "${SOURCE}" NAME_WE)
    set(OBJECT "${CMAKE_CURRENT_BINARY_DIR}/${NAME}.o")
    set(EMBEDDED "${CMAKE_CURRENT_BINARY_DIR}/${NAME}_embedded.o")
    set(EXTRA_HEADERS)
    if(NAME STREQUAL "deepseek_v41_attention_norm_bf16_gaudi2" OR
       NAME STREQUAL "deepseek_v41_qnorm_quant_gaudi2" OR
       NAME STREQUAL "deepseek_v41_kv_norm_rope_bf16_gaudi2")
        list(APPEND EXTRA_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/../flashinfer_gaudi/kernels/norm_math_gaudi2.h")
    elseif(NAME STREQUAL "deepseek_v41_mla_shared_kv_gaudi2")
        list(APPEND EXTRA_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/deepseek_v41_mla_gather_gaudi2.c")
    elseif(NAME STREQUAL "deepseek_v41_mla_exp_bf16_gaudi2")
        list(APPEND EXTRA_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/deepseek_v41_mla_softmax_gaudi2.c")
    endif()
    add_custom_command(OUTPUT "${OBJECT}"
        COMMAND "${TPC_COMPILER}" -Wall -Werror -O2 -march=gaudi2
            -I${TPC_INCLUDE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/include
            "${SOURCE}" -c -o "${OBJECT}"
        DEPENDS "${SOURCE}" ${KERNEL_HEADERS} ${EXTRA_HEADERS} VERBATIM)
    add_custom_command(OUTPUT "${EMBEDDED}"
        COMMAND "${ELF_OBJCOPY}" -I binary -O elf64-x86-64 -B i386:x86-64
            "./${NAME}.o" "${NAME}_embedded.o"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        DEPENDS "${OBJECT}" VERBATIM)
    set_source_files_properties("${EMBEDDED}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
    list(APPEND EMBEDDED_OBJECTS "${EMBEDDED}")
endforeach()
file(GLOB HOST_SOURCES CONFIGURE_DEPENDS "host/*.cpp")
add_library(deepseek_v4_gaudi2_kernels SHARED ${HOST_SOURCES} ${EMBEDDED_OBJECTS})
target_include_directories(deepseek_v4_gaudi2_kernels PRIVATE
    host "${HABANA_INCLUDE_DIR}" "${TPC_INCLUDE_DIR}")
target_compile_options(deepseek_v4_gaudi2_kernels PRIVATE -Wall -Wextra -fno-strict-aliasing)
target_link_options(deepseek_v4_gaudi2_kernels PRIVATE -Wl,-z,noexecstack -Wl,--no-undefined)
target_link_libraries(deepseek_v4_gaudi2_kernels PRIVATE ${CMAKE_DL_LIBS})
enable_testing()
add_executable(deepseek_v4_database_test tests/kernel_database.cpp)
target_include_directories(deepseek_v4_database_test PRIVATE "${HABANA_INCLUDE_DIR}" "${TPC_INCLUDE_DIR}")
target_compile_options(deepseek_v4_database_test PRIVATE -UNDEBUG)
target_link_libraries(deepseek_v4_database_test PRIVATE deepseek_v4_gaudi2_kernels)
add_test(NAME deepseek_v4_database COMMAND deepseek_v4_database_test)
