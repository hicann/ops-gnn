"""
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
"""
import os
import shutil
import subprocess
import warnings

from setuptools import Distribution, find_packages, setup
from setuptools.command.build_py import build_py

__version__ = '0.1.0'
URL = 'https://gitcode.com/cann/ops-gnn'

BUILD_DOCS = os.getenv('BUILD_DOCS', '0') == '1'


def _warn_default_npu_arch(reason):
    warnings.warn(
        f'由于{reason}，当前默认以 Ascend 950 设备进行编译'
        '（dav-3510 / arch35）。'
        '可设置 NPU_ARCH 环境变量显式指定目标架构。',
        RuntimeWarning,
        stacklevel=2,
    )


def detect_npu_arch():
    """Return the CMake NPU_ARCH for the locally visible Ascend product.

    按芯片型号识别：A2(910B)/A3(910C)→dav-2201，950→dav-3510，其他报错不支持。
    """
    npu_smi = shutil.which('npu-smi')
    if npu_smi is None:
        _warn_default_npu_arch('检测不到 NPU 设备（未找到 npu-smi）')
        return 'dav-3510'
    try:
        result = subprocess.run(
            [npu_smi, 'info', '-m'], check=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, timeout=10)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        _warn_default_npu_arch(
            f'检测不到 NPU 设备（npu-smi info 执行失败：{exc}）')
        return 'dav-3510'
    product_info = result.stdout.lower()
    if '910_93' in product_info or '910c' in product_info or '910b' in product_info:
        return 'dav-2201'
    if '950' in product_info:
        return 'dav-3510'
    raise RuntimeError(
        '不支持的芯片型号，仅支持 950 / A2(910B) / A3(910C)。'
        '可设置 NPU_ARCH=dav-3510 或 NPU_ARCH=dav-2201。')

def build_with_cmake():
    if 'ASCEND_HOME_PATH' not in os.environ:
        raise EnvironmentError("ASCEND_HOME_PATH environment variable not set. Please source set_env.sh first.")
    
    npu_arch = os.getenv('NPU_ARCH') or os.getenv('TARGET_NPU_ARCH')
    if npu_arch is None:
        npu_arch = detect_npu_arch()
    if npu_arch not in ('dav-3510', 'dav-2201'):
        raise ValueError(f'Unsupported NPU_ARCH: {npu_arch}')
    print(f'NPU 架构: {npu_arch}')
    cmake_build_dir = f'build/cmake_python_{npu_arch}'
    # 删除 CMakeCache.txt，避免 pip 拷贝源码到临时目录后
    # cmake 检测到缓存的源路径与当前路径不一致而报错
    cache_file = os.path.join(cmake_build_dir, 'CMakeCache.txt')
    if os.path.exists(cache_file):
        os.remove(cache_file)
    os.makedirs(cmake_build_dir, exist_ok=True)
    
    cmake_cmd = [
        'cmake',
        '-DWITH_PYTHON=ON',
        '-DCMAKE_BUILD_TYPE=Release',
        f'-DNPU_ARCH={npu_arch}',
        '-S', '.',
        '-B', cmake_build_dir,
    ]
    
    print(f'Running CMake: {" ".join(cmake_cmd)}')
    subprocess.run(cmake_cmd, check=True)
    
    make_cmd = ['make', '-C', cmake_build_dir, '-j4']
    print(f'Running Make: {" ".join(make_cmd)}')
    subprocess.run(make_cmd, check=True)
    
    npu_kernel_lib = os.path.join('output', 'kernel', 'libopsgnn_npu_kernel.so')
    if os.path.exists(npu_kernel_lib):
        print(f'NPU kernel library generated at: {npu_kernel_lib}')
    
    pybind_lib = os.path.join(cmake_build_dir, 'lib_pybind.so')
    if os.path.exists(pybind_lib):
        pybind_dest = os.path.join('python', 'ops_gnn', '_pybind.so')
        shutil.copy2(pybind_lib, pybind_dest)
        print(f'Copied _pybind.so to: {pybind_dest}')

if not BUILD_DOCS:
    build_with_cmake()

class CustomBuildPy(build_py):
    def run(self):
        build_py.run(self)

        for library_name in ('_pybind.so', 'libopsgnn_npu_kernel.so'):
            library_src = os.path.join('python', 'ops_gnn', library_name)
            if os.path.exists(library_src):
                library_dest = os.path.join(self.build_lib, 'ops_gnn', library_name)
                os.makedirs(os.path.dirname(library_dest), exist_ok=True)
                shutil.copy2(library_src, library_dest)
                print(f'Copied {library_name} to build lib: {library_dest}')


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

setup(
    name='ops_gnn',
    version=__version__,
    description='OpsGNN: Library of Optimized Graph Neural Network Algorithms for NPU',
    license='MIT',
    author='Ascend',
    url=URL,
    download_url=f'{URL}/archive/{__version__}.tar.gz',
    python_requires='>=3.8',
    ext_modules=[],
    distclass=BinaryDistribution,
    cmdclass={
        'build_py': CustomBuildPy,
    },
    packages=find_packages('python'),
    package_dir={'': 'python'},
    package_data={'ops_gnn': ['*.so']},
    include_package_data=True,
)
