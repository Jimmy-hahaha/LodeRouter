cmake_minimum_required(VERSION 3.20)

project(LodeRouter
    VERSION 1.0.0
    DESCRIPTION "A difficulty-aware LLM model router"
    LANGUAGES CXX
)

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        Debug Release RelWithDebInfo MinSizeRel)
endif()

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
set(LODEROUTER_LINKER "auto" CACHE STRING
    "Linker backend: auto | mold | default")
set_property(CACHE LODEROUTER_LINKER PROPERTY STRINGS auto mold default)

option(LODEROUTER_USE_PCH "Enable precompiled headers" ON)

# ---------------------------------------------------------------------------
# Linker selection (mold with graceful fallback)
# ---------------------------------------------------------------------------
set(_loderouter_use_mold FALSE)

if(LODEROUTER_LINKER STREQUAL "mold")
    find_program(MOLD_EXECUTABLE mold)
    if(NOT MOLD_EXECUTABLE)
        message(FATAL_ERROR
            "LODEROUTER_LINKER=mold requested, but mold was not found. "
            "Install mold or configure with -DLODEROUTER_LINKER=default.")
    endif()
    set(_loderouter_use_mold TRUE)
    message(STATUS "Linker: mold (forced) -> ${MOLD_EXECUTABLE}")

elseif(LODEROUTER_LINKER STREQUAL "auto")
    if(UNIX AND NOT APPLE)
        find_program(MOLD_EXECUTABLE mold)
        if(MOLD_EXECUTABLE)
            set(_loderouter_use_mold TRUE)
            message(STATUS "Linker: mold (auto)   -> ${MOLD_EXECUTABLE}")
        else()
            message(STATUS "Linker: system default (mold not found)")
        endif()
    else()
        message(STATUS "Linker: system default (mold unsupported on this platform)")
    endif()

elseif(LODEROUTER_LINKER STREQUAL "default")
    message(STATUS "Linker: system default (requested)")

else()
    message(FATAL_ERROR
        "Unknown LODEROUTER_LINKER='${LODEROUTER_LINKER}'. "
        "Valid values: auto | mold | default.")
endif()

if(_loderouter_use_mold)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.29")
        set(CMAKE_LINKER_TYPE MOLD)
    else()
        add_link_options("-fuse-ld=mold")
    endif()
endif()

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
add_executable(LodeRouter
    src/main.cpp
)

add_executable(config_tool
    src/config_tool.cpp
)

# ---------------------------------------------------------------------------
# Precompiled headers (opt-in)
# ---------------------------------------------------------------------------
if(LODEROUTER_USE_PCH)
    target_precompile_headers(LodeRouter
        PRIVATE
            <iostream>
            <string>
            <string_view>
            <vector>
            <memory>
            <unordered_map>
            <optional>
            <functional>
    )
    message(STATUS "PCH:     enabled")
else()
    message(STATUS "PCH:     disabled")
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(fmt CONFIG QUIET)

foreach(_target LodeRouter config_tool)
    target_include_directories(${_target}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(${_target}
        PRIVATE
            Threads::Threads
            OpenSSL::SSL
            OpenSSL::Crypto
            $<$<BOOL:${fmt_FOUND}>:fmt::fmt>
    )
endforeach()

# ---------------------------------------------------------------------------
# Laya judge backend (sibling project: ../laya)
# ---------------------------------------------------------------------------
# 裁判模型用同目录的 laya 项目（本地 ONNX 推理），默认 ../laya/cpp。
set(LAYA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../laya/cpp" CACHE PATH
    "Directory of the laya C++ project (contains CMakeLists.txt, laya.cpp, laya.h)")

if(NOT EXISTS "${LAYA_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "laya project not found at '${LAYA_DIR}'. "
        "Clone it next to LodeRouter or configure with -DLAYA_DIR=<path to laya/cpp>.")
endif()

add_subdirectory("${LAYA_DIR}" "${CMAKE_BINARY_DIR}/laya" EXCLUDE_FROM_ALL)

# 构建时记录的 laya 模型路径，运行时找不到 judge_model_dir 时作为自动探测候选。
set(LAYA_MODEL_DIR "${LAYA_DIR}/../laya-zh-onnx-int8" CACHE PATH
    "laya ONNX model directory used as the build-time default for auto-detection")
get_filename_component(LAYA_MODEL_DIR "${LAYA_MODEL_DIR}" ABSOLUTE)

target_link_libraries(LodeRouter PRIVATE laya)
target_compile_definitions(LodeRouter PRIVATE
    LODEROUTER_LAYA_MODEL_DIR="${LAYA_MODEL_DIR}"
)

message(STATUS "laya:    ${LAYA_DIR}")
message(STATUS "laya model default: ${LAYA_MODEL_DIR}")
