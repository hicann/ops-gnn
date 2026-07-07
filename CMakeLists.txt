cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")

set(VALID_NPU_ARCHS dav-3510 dav-2201)
if(NOT DEFINED NPU_ARCH OR NPU_ARCH STREQUAL "")
    set(NPU_ARCH "dav-3510" CACHE STRING "NPU architecture")
endif()
if(NOT "${NPU_ARCH}" IN_LIST VALID_NPU_ARCHS)
    message(FATAL_ERROR "Unsupported NPU_ARCH: ${NPU_ARCH}. Supported values: dav-3510, dav-2201")
endif()

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

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH environment variable not set. Please source /usr/local/Ascend/cann-9.1.0-beta.1/bin/setenv.bash first.")
endif()
set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")

set(CMAKE_C_COMPILER gcc CACHE FILEPATH "C compiler" FORCE)
set(CMAKE_CXX_COMPILER g++ CACHE FILEPATH "CXX compiler" FORCE)

set(BISHENG "${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/ccec_compiler/bin/bisheng" CACHE FILEPATH "Path to Bisheng compiler")
message(STATUS "Bisheng compiler path: ${BISHENG}")
set(CMAKE_ASC_COMPILER "${BISHENG}" CACHE FILEPATH "ASC compiler" FORCE)

set(CMAKE_PREFIX_PATH ${ASCEND_HOME_PATH}/ ${CMAKE_PREFIX_PATH})
find_package(ASC REQUIRED)

project(opsgnn LANGUAGES CXX ASC)

set(OPSGNN_VERSION 0.1.0)

set(ASCEND_C_INCLUDE_DIRS
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

include_directories(${ASCEND_C_INCLUDE_DIRS})
include_directories(csrc)
include_directories(csrc/npu)
include_directories(csrc/npu/kernel)

option(WITH_PYTHON "Link to Python when building" ON)

if(WITH_PYTHON)
    add_definitions(-DWITH_PYTHON)
    find_package(Python3 COMPONENTS Development REQUIRED)
endif()

if(DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}" "${CMAKE_PREFIX_PATH}")
endif()

if(DEFINED ENV{VIRTUAL_ENV})
    set(CMAKE_PREFIX_PATH "$ENV{VIRTUAL_ENV}" "${CMAKE_PREFIX_PATH}")
endif()

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

find_package(Torch REQUIRED)

set(OPERATOR_SOURCES
    csrc/pybind.cpp
    csrc/npu/add_sample.cpp
)

set(NPU_KERNEL_LIB "")

file(GLOB NPU_ASC_FILES csrc/npu/kernel/*.asc)

if(NPU_ASC_FILES)
    set(NPU_KERNEL_LIB_NAME "opsgnn_npu_kernel")
    set(OUTPUT_KERNEL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/output/kernel)
    set(NPU_KERNEL_LIB ${OUTPUT_KERNEL_DIR}/lib${NPU_KERNEL_LIB_NAME}.so)
    
    set(NPU_KERNEL_OBJECTS "")
    foreach(ASC_FILE ${NPU_ASC_FILES})
        get_filename_component(FILE_NAME ${ASC_FILE} NAME_WE)
        set(OBJ_FILE ${CMAKE_CURRENT_BINARY_DIR}/${FILE_NAME}.o)
        list(APPEND NPU_KERNEL_OBJECTS ${OBJ_FILE})
        
        add_custom_command(
            OUTPUT ${OBJ_FILE}
            COMMAND ${CMAKE_ASC_COMPILER}
            ARGS -c ${ASC_FILE}
                --npu-arch=${NPU_ARCH}
                -O2
                -fPIC
                "-I${ASCEND_HOME_PATH}/include/ascendc/basic_api"
                "-I${ASCEND_HOME_PATH}/include/ascendc"
                "-I${ASCEND_HOME_PATH}/include"
                "-I${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/include/simt_api"
                "-I${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/include"
                "-I${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/include/utils/base"
                "-I${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/include/tiling"
                "-I${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/include/tiling/platform"
                "-I${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/include/interface"
                "-I${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/include/basic_api"
                "-I${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/ascendc/include/basic_api"
                "-I${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/ascendc/include/basic_api/impl"
                "-I${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/asc/impl/basic_api"
                "-I${ASCEND_HOME_PATH}/acllib/include"
                "-I${CMAKE_CURRENT_SOURCE_DIR}/csrc/npu/kernel"
                -o ${OBJ_FILE}
            MAIN_DEPENDENCY ${ASC_FILE}
            COMMENT "Compiling ASC kernel: ${FILE_NAME}.asc"
        )
    endforeach()
    
    add_custom_command(
        OUTPUT ${NPU_KERNEL_LIB}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_KERNEL_DIR}
        COMMAND ${CMAKE_ASC_COMPILER}
        ARGS -shared -fPIC -o ${NPU_KERNEL_LIB}
            ${NPU_KERNEL_OBJECTS}
            -L${ASCEND_HOME_PATH}/lib64
            -L${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/ccec_compiler/lib
            -L${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/lib64
            -L/lib/aarch64-linux-gnu
            -Wl,-rpath,${ASCEND_HOME_PATH}/lib64
            -Wl,-rpath,${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/ccec_compiler/lib
            -Wl,-rpath,${ASCEND_HOME_PATH}/${SYSTEM_PREFIX}/lib64
            -lplatform
            -lascendcl
            -lruntime
            -lascendc_runtime
            -ltiling_api
            -lunified_dlog
            -ldl
            -lstdc++
            -lm
        MAIN_DEPENDENCY ${NPU_KERNEL_OBJECTS}
        COMMENT "Linking NPU kernel library: ${NPU_KERNEL_LIB}"
    )
    
    add_custom_target(npu_kernel_target ALL DEPENDS ${NPU_KERNEL_LIB})
endif()

add_library(_pybind SHARED ${OPERATOR_SOURCES})
target_compile_definitions(_pybind PRIVATE 
    WITH_PYTHON WITH_NPU AT_PARALLEL_OPENMP 
    TORCH_API_INCLUDE_EXTENSION_H TORCH_EXTENSION_NAME=_pybind)
target_link_libraries(_pybind PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(_pybind PRIVATE ${TORCH_INSTALL_PREFIX}/lib/libtorch_python.so)

if(WITH_PYTHON)
    target_link_libraries(_pybind PRIVATE Python3::Python)
endif()

if(NPU_KERNEL_LIB)
    add_dependencies(_pybind npu_kernel_target)
    target_link_libraries(_pybind PRIVATE ${NPU_KERNEL_LIB})
endif()

target_link_directories(_pybind PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/acllib/lib64
)

target_link_libraries(_pybind PRIVATE
    platform
    ascendcl
    runtime
    tiling_api
    unified_dlog
    dl
)

set_target_properties(_pybind PROPERTIES 
    EXPORT_NAME OpsGNN
    LINK_FLAGS "-s"
    LIBRARY_OUTPUT_NAME "_pybind"
)

target_include_directories(_pybind PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/csrc>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}>
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}/pybind11)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/output CACHE PATH "Install path prefix" FORCE)

set(OPSGNN_CMAKECONFIG_INSTALL_DIR "share/cmake/OpsGNN" CACHE STRING "install path for OpsGNNConfig.cmake")

configure_package_config_file(cmake/OpsGNNConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/OpsGNNConfig.cmake"
    INSTALL_DESTINATION ${OPSGNN_CMAKECONFIG_INSTALL_DIR})

write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/OpsGNNConfigVersion.cmake
    VERSION ${OPSGNN_VERSION}
    COMPATIBILITY AnyNewerVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/OpsGNNConfig.cmake
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