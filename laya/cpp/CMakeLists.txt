cmake_minimum_required(VERSION 3.20)

project(laya
    VERSION 1.0.0
    DESCRIPTION "Pure C++ runtime for the laya decision model (ONNX)"
    LANGUAGES CXX
)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(LAYA_IS_TOP_LEVEL ON)
else()
    set(LAYA_IS_TOP_LEVEL OFF)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LAYA_BUILD_CLI "Build the laya-cli tool" ${LAYA_IS_TOP_LEVEL})
option(LAYA_BUILD_TESTS "Build the laya C++ smoke tests" ${LAYA_IS_TOP_LEVEL})

find_package(Threads REQUIRED)

# ───────── ONNX Runtime ─────────
# Prefer the package config exported by system/pip installations, then fall
# back to a manual header/library probe (e.g. onnxruntime in /usr/local).
find_package(onnxruntime CONFIG QUIET)

if(NOT TARGET onnxruntime::onnxruntime)
    find_path(LAYA_ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        HINTS
            ${LAYA_ONNXRUNTIME_ROOT}
            $ENV{LAYA_ONNXRUNTIME_ROOT}
            /usr/local/include/onnxruntime
            /usr/include/onnxruntime
            /opt/homebrew/include/onnxruntime
    )
    find_library(LAYA_ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        HINTS
            ${LAYA_ONNXRUNTIME_ROOT}
            $ENV{LAYA_ONNXRUNTIME_ROOT}
            /usr/local/lib
            /usr/lib
            /opt/homebrew/lib
    )
    if(NOT LAYA_ONNXRUNTIME_INCLUDE_DIR OR NOT LAYA_ONNXRUNTIME_LIBRARY)
        message(FATAL_ERROR
            "ONNX Runtime was not found. Install the onnxruntime development "
            "package (onnxruntime_cxx_api.h + libonnxruntime) or configure with "
            "-DLAYA_ONNXRUNTIME_ROOT=<prefix>.")
    endif()
    add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED)
    set_target_properties(onnxruntime::onnxruntime PROPERTIES
        IMPORTED_LOCATION "${LAYA_ONNXRUNTIME_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${LAYA_ONNXRUNTIME_INCLUDE_DIR}"
    )
    message(STATUS "laya: onnxruntime (manual) -> ${LAYA_ONNXRUNTIME_LIBRARY}")
else()
    message(STATUS "laya: onnxruntime (package) -> found")
endif()

# ───────── Library ─────────
add_library(laya STATIC
    laya.cpp
    tokenizer.cpp
    onnx_session.cpp
)

target_include_directories(laya
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

target_link_libraries(laya
    PUBLIC
        Threads::Threads
    PRIVATE
        onnxruntime::onnxruntime
)

if(MSVC)
    target_compile_options(laya PRIVATE /W4)
else()
    target_compile_options(laya PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ───────── CLI ─────────
if(LAYA_BUILD_CLI)
    add_executable(laya-cli cli.cpp)
    target_link_libraries(laya-cli PRIVATE laya)
    target_include_directories(laya-cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
    set_target_properties(laya-cli PROPERTIES OUTPUT_NAME laya-cli)
endif()

# ───────── Tests ─────────
if(LAYA_BUILD_TESTS)
    enable_testing()

    add_executable(laya-engine-smoke tests/engine_smoke.cpp)
    target_link_libraries(laya-engine-smoke PRIVATE laya)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../model.onnx")
        add_test(NAME engine_smoke COMMAND laya-engine-smoke ${CMAKE_CURRENT_SOURCE_DIR}/..)
    endif()

    add_executable(laya-onnx-smoke tests/onnx_session_smoke.cpp)
    target_link_libraries(laya-onnx-smoke PRIVATE laya)
    target_include_directories(laya-onnx-smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../model.onnx")
        add_test(NAME onnx_session_smoke COMMAND laya-onnx-smoke ${CMAKE_CURRENT_SOURCE_DIR}/..)
    endif()

    add_executable(laya-tokenizer-parity tests/tokenizer_parity.cpp)
    target_link_libraries(laya-tokenizer-parity PRIVATE laya)
    target_include_directories(laya-tokenizer-parity PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
    set(LAYA_TOKENIZER_CORPUS
        "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/laya_oracle/reference/tokenizer_corpus.jsonl")
    if(EXISTS "${LAYA_TOKENIZER_CORPUS}")
        add_test(NAME tokenizer_parity COMMAND laya-tokenizer-parity
            "${CMAKE_CURRENT_SOURCE_DIR}/../tokenizer.json" "${LAYA_TOKENIZER_CORPUS}")
    endif()
endif()
