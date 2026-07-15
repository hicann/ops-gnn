# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")

set(OPSGNN_VERSION 0.1.0)

###############################################################################
# Configuration Options
###############################################################################

set(VALID_NPU_ARCHS dav-3510 dav-2201)
if(NOT DEFINED NPU_ARCH OR NPU_ARCH STREQUAL "")
    set(NPU_ARCH "dav-3510" CACHE STRING "NPU architecture")
endif()
if(NOT "${NPU_ARCH}" IN_LIST VALID_NPU_ARCHS)
    message(FATAL_ERROR "Unsupported NPU_ARCH: ${NPU_ARCH}. Supported values: dav-3510, dav-2201")
endif()

option(WITH_PYTHON "Link to Python when building" ON)

###############################################################################
# System Detection
###############################################################################

if(UNIX)
    set(_ascend_processor "${CMAKE_SYSTEM_PROCESSOR}")
    if(NOT _ascend_processor)
        execute_process(
            COMMAND uname -m
            OUTPUT_VARIABLE _ascend_processor
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    endif()
    set(SYSTEM_PREFIX "${_ascend_processor}-linux")
endif()

###############################################################################
# Ascend/CANN Environment
###############################################################################

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH environment variable not set. Please source /usr/local/Ascend/cann-9.1.0-beta.1/bin/setenv.bash first.")
endif()
set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")

set(BISHENG "${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/ccec_compiler/bin/bisheng" CACHE FILEPATH "Path to Bisheng compiler")
message(STATUS "Bisheng compiler path: ${BISHENG}")
set(CMAKE_ASC_COMPILER "${BISHENG}" CACHE FILEPATH "ASC compiler" FORCE)

set(CMAKE_PREFIX_PATH ${ASCEND_HOME_PATH} ${CMAKE_PREFIX_PATH})
find_package(ASC REQUIRED)

project(opsgnn LANGUAGES CXX ASC)

set(ASCEND_INCLUDE_DIRS
    ${ASCEND_HOME_PATH}/include/ascendc/basic_api
    ${ASCEND_HOME_PATH}/include/ascendc
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/include/simt_api
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/include
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/include/utils/base
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/include/tiling
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/include/tiling/platform
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/include/interface
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/include/basic_api
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/ascendc/include/basic_api
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/ascendc/include/basic_api/impl
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/impl/basic_api
    ${ASCEND_HOME_PATH}/acllib/include
)

set(ASCEND_LINK_DIRS
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/ccec_compiler/lib
    ${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/lib64
    ${ASCEND_HOME_PATH}/acllib/lib64
)

set(ASCEND_LIBRARIES
    platform
    ascendcl
    runtime
    ascendc_runtime
    tiling_api
    unified_dlog
    dl
    stdc++
    m
)

###############################################################################
# Python3 Configuration
###############################################################################

if(WITH_PYTHON)
    if(DEFINED ENV{Python3_ROOT_DIR})
        set(Python3_ROOT_DIR "$ENV{Python3_ROOT_DIR}" CACHE PATH "Python3 root directory")
        message(STATUS "Python3_ROOT_DIR set from environment: ${Python3_ROOT_DIR}")

        execute_process(
            COMMAND "${Python3_ROOT_DIR}/bin/python3" -c "import sysconfig; print(sysconfig.get_paths()['include'])"
            OUTPUT_VARIABLE Python3_INCLUDE_DIRS
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        execute_process(
            COMMAND "${Python3_ROOT_DIR}/bin/python3" -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
            OUTPUT_VARIABLE Python3_LIBDIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        execute_process(
            COMMAND "${Python3_ROOT_DIR}/bin/python3" -c "import sysconfig; print(sysconfig.get_config_var('LDLIBRARY'))"
            OUTPUT_VARIABLE Python3_LDLIBRARY
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        set(Python3_LIBRARIES "${Python3_LIBDIR}/${Python3_LDLIBRARY}")
        set(Python3_FOUND TRUE)

        add_library(Python3::Python SHARED IMPORTED)
        set_property(TARGET Python3::Python PROPERTY IMPORTED_LOCATION "${Python3_LIBRARIES}")
        set_property(TARGET Python3::Python PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${Python3_INCLUDE_DIRS}")

        message(STATUS "Manually detected Python3:")
        message(STATUS "  Include: ${Python3_INCLUDE_DIRS}")
        message(STATUS "  Library: ${Python3_LIBRARIES}")
    else()
        find_package(Python3 COMPONENTS Development QUIET)
        if(NOT Python3_FOUND)
            message(FATAL_ERROR "Python3 not found. Please set Python3_ROOT_DIR environment variable.")
        endif()
    endif()
endif()

###############################################################################
# PyTorch Configuration
###############################################################################

execute_process(
    COMMAND python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE TORCH_INSTALL_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

execute_process(
    COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(TORCH_INSTALL_PREFIX)
    set(ENV{TORCH_INSTALL_PREFIX} "${TORCH_INSTALL_PREFIX}")
endif()

if(TORCH_CMAKE_PREFIX_PATH)
    set(CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}" "${CMAKE_PREFIX_PATH}")
endif()

# Suppress kineto warning for CPU torch: pre-set to an existing torch lib
# so find_library skips the search and no "not found" warning is emitted
set(kineto_LIBRARY "${TORCH_INSTALL_PREFIX}/lib/libtorch.so" CACHE FILEPATH "" FORCE)
find_package(Torch REQUIRED)

message(STATUS "Found PyTorch: ${TORCH_INSTALL_PREFIX}")

###############################################################################
# Source Files
###############################################################################

file(GLOB_RECURSE NPU_ASC_FILES 
    csrc/npu/kernel/*.asc 
    csrc/npu/kernel/*.cpp
)

file(GLOB_RECURSE NPU_CPP_INTERFACE_FILES 
    csrc/npu/host/*.cpp
)

set(OPERATOR_SOURCES
    csrc/pybind.cpp
    ${NPU_CPP_INTERFACE_FILES}
)

###############################################################################
# NPU Kernel Compilation
###############################################################################

set(NPU_KERNEL_LIB "")

if(NPU_ASC_FILES)
    set(NPU_KERNEL_LIB_NAME "opsgnn_npu_kernel")
    set(OUTPUT_KERNEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output/kernel")
    set(NPU_KERNEL_LIB "${OUTPUT_KERNEL_DIR}/lib${NPU_KERNEL_LIB_NAME}.so")

    set(NPU_KERNEL_OBJECTS "")
    foreach(ASC_FILE ${NPU_ASC_FILES})
        get_filename_component(FILE_NAME ${ASC_FILE} NAME_WE)
        get_filename_component(FILE_EXT ${ASC_FILE} EXT)
        set(OBJ_FILE "${CMAKE_CURRENT_BINARY_DIR}/${FILE_NAME}.o")
        list(APPEND NPU_KERNEL_OBJECTS ${OBJ_FILE})

        set(ASC_LANG_FLAG "")
        if("${FILE_EXT}" STREQUAL ".cpp")
            set(ASC_LANG_FLAG "--asc-aicore-lang")
        endif()

        set(ASC_INCLUDE_FLAGS "")
        foreach(INC_DIR ${ASCEND_INCLUDE_DIRS})
            list(APPEND ASC_INCLUDE_FLAGS "-I${INC_DIR}")
        endforeach()
        list(APPEND ASC_INCLUDE_FLAGS 
            "-I${CMAKE_CURRENT_SOURCE_DIR}/csrc"
            "-I${CMAKE_CURRENT_SOURCE_DIR}/csrc/npu"
            "-I${CMAKE_CURRENT_SOURCE_DIR}/csrc/npu/kernel"
        )

        add_custom_command(
            OUTPUT ${OBJ_FILE}
            COMMAND ${CMAKE_ASC_COMPILER}
            ARGS ${ASC_LANG_FLAG}
                -c ${ASC_FILE}
                --npu-arch=${NPU_ARCH}
                -O2
                -fPIC
                ${ASC_INCLUDE_FLAGS}
                -o ${OBJ_FILE}
            MAIN_DEPENDENCY ${ASC_FILE}
            COMMENT "Compiling ASC kernel: ${FILE_NAME}${FILE_EXT}"
        )
    endforeach()

    set(ASC_LINK_FLAGS "")
    foreach(LINK_DIR ${ASCEND_LINK_DIRS})
        list(APPEND ASC_LINK_FLAGS "-L${LINK_DIR}")
    endforeach()
    foreach(LINK_DIR ${ASCEND_LINK_DIRS})
        list(APPEND ASC_LINK_FLAGS "-Wl,-rpath,${LINK_DIR}")
    endforeach()
    foreach(LIB_NAME ${ASCEND_LIBRARIES})
        list(APPEND ASC_LINK_FLAGS "-l${LIB_NAME}")
    endforeach()

    add_custom_command(
        OUTPUT ${NPU_KERNEL_LIB}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_KERNEL_DIR}
        COMMAND ${CMAKE_ASC_COMPILER}
        ARGS -shared -fPIC -o ${NPU_KERNEL_LIB}
            ${NPU_KERNEL_OBJECTS}
            ${ASC_LINK_FLAGS}
        DEPENDS ${NPU_KERNEL_OBJECTS}
        COMMENT "Linking NPU kernel library: ${NPU_KERNEL_LIB}"
    )

    add_custom_target(npu_kernel_target ALL DEPENDS ${NPU_KERNEL_LIB})
endif()

###############################################################################
# Main Library Target
###############################################################################

add_library(_pybind SHARED ${OPERATOR_SOURCES})

target_compile_definitions(_pybind PRIVATE 
    WITH_PYTHON WITH_NPU AT_PARALLEL_OPENMP 
    TORCH_API_INCLUDE_EXTENSION_H TORCH_EXTENSION_NAME=_pybind
)

target_include_directories(_pybind PRIVATE
    ${ASCEND_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/npu
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/npu/host
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/npu/kernel
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}/pybind11
)

if(WITH_PYTHON)
    target_include_directories(_pybind PRIVATE ${Python3_INCLUDE_DIRS})
    target_link_libraries(_pybind PRIVATE Python3::Python)
endif()

target_link_libraries(_pybind PRIVATE
    ${TORCH_LIBRARIES}
    ${TORCH_INSTALL_PREFIX}/lib/libtorch_python.so
)

target_link_directories(_pybind PRIVATE ${ASCEND_LINK_DIRS})
target_link_libraries(_pybind PRIVATE ${ASCEND_LIBRARIES})

if(NPU_KERNEL_LIB)
    add_dependencies(_pybind npu_kernel_target)
    target_link_libraries(_pybind PRIVATE ${NPU_KERNEL_LIB})
endif()

set_target_properties(_pybind PROPERTIES 
    EXPORT_NAME OpsGNN
    LINK_FLAGS "-s"
    LIBRARY_OUTPUT_NAME "_pybind"
)

###############################################################################
# Installation
###############################################################################

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/output" CACHE PATH "Install path prefix" FORCE)
set(OPSGNN_CMAKECONFIG_INSTALL_DIR "share/cmake/OpsGNN" CACHE STRING "install path for OpsGNNConfig.cmake")

configure_package_config_file(cmake/OpsGNNConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/OpsGNNConfig.cmake"
    INSTALL_DESTINATION ${OPSGNN_CMAKECONFIG_INSTALL_DIR})

write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/OpsGNNConfigVersion.cmake
    VERSION ${OPSGNN_VERSION}
    COMPATIBILITY AnyNewerVersion)

install(FILES 
    ${CMAKE_CURRENT_BINARY_DIR}/OpsGNNConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/OpsGNNConfigVersion.cmake
    DESTINATION ${OPSGNN_CMAKECONFIG_INSTALL_DIR})

install(TARGETS _pybind
    EXPORT OpsGNNTargets
    LIBRARY DESTINATION kernel
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

if(NPU_KERNEL_LIB)
    install(FILES ${NPU_KERNEL_LIB} DESTINATION kernel
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
endif()

install(EXPORT OpsGNNTargets
    NAMESPACE OpsGNN::
    DESTINATION ${OPSGNN_CMAKECONFIG_INSTALL_DIR})
