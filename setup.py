"""
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
"""

import glob
import os
import os.path as osp
import platform
import shutil
import subprocess
import sys

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel

__version__ = '0.1.0'
URL = 'https://gitcode.com/cann/ops-gnn'

BUILD_DOCS = os.getenv('BUILD_DOCS', '0') == '1'


def compile_npu_kernel():
    import torch
    
    if 'ASCEND_HOME_PATH' not in os.environ:
        raise EnvironmentError("ASCEND_HOME_PATH environment variable not set. Please source /usr/local/Ascend/cann-9.1.0-beta.1/bin/setenv.bash first.")
    
    cann_path = os.environ['ASCEND_HOME_PATH']
    
    asc_files = glob.glob('csrc/npu/kernel/*.asc')
    if not asc_files:
        return None
    
    npu_kernel_lib = 'build/kernel/libopsgnn_npu_kernel.so'
    os.makedirs(os.path.dirname(npu_kernel_lib), exist_ok=True)
    
    include_flags = [
        f'-I{cann_path}/include/ascendc/basic_api',
        f'-I{cann_path}/include/ascendc',
        f'-I{cann_path}/include',
        f'-I{cann_path}/x86_64-linux/asc/include/simt_api',
        f'-I{cann_path}/x86_64-linux/asc/include',
        f'-I{cann_path}/x86_64-linux/asc/include/utils/base',
        f'-I{cann_path}/acllib/include',
        '-Icsrc',
        '-Icsrc/npu',
        '-Icsrc/npu/kernel',
    ]
    
    link_flags = [
        f'-L{cann_path}/lib64',
        f'-L{cann_path}/acllib/lib64',
        '-lplatform',
        '-lascendcl',
        '-lruntime',
        '-ltiling_api',
        '-lunified_dlog',
        '-ldl',
    ]
    
    cmd = ['ccec', '--asc-aicore-lang', '--npu-arch=dav-3510', '-shared', '-o', npu_kernel_lib, '-std=c++17', '-D_GLIBCXX_USE_CXX11_ABI=1', '-fPIC', '-O2'] + include_flags + asc_files + link_flags
    
    print(f'Compiling NPU kernel library: {" ".join(cmd)}')
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(f'Stdout: {result.stdout}')
        
        output_kernel_dir = 'output/kernel'
        os.makedirs(output_kernel_dir, exist_ok=True)
        output_kernel_lib = os.path.join(output_kernel_dir, os.path.basename(npu_kernel_lib))
        shutil.copy2(npu_kernel_lib, output_kernel_lib)
        print(f'Copied NPU kernel library to: {output_kernel_lib}')
        
        return npu_kernel_lib
    except subprocess.CalledProcessError as e:
        print(f'NPU kernel compilation failed: {e}')
        print(f'Stderr: {e.stderr if e.stderr else "None"}')
        print(f'Stdout: {e.stdout if e.stdout else "None"}')
        raise


def build_with_cmake():
    import torch
    
    if 'ASCEND_HOME_PATH' not in os.environ:
        raise EnvironmentError("ASCEND_HOME_PATH environment variable not set. Please source /usr/local/Ascend/cann-9.1.0-beta.1/bin/setenv.bash first.")
    
    cann_path = os.environ['ASCEND_HOME_PATH']
    
    cmake_build_dir = 'build/cmake_python'
    os.makedirs(cmake_build_dir, exist_ok=True)
    
    cmake_cmd = [
        'cmake',
        '-DWITH_PYTHON=ON',
        '-DCMAKE_BUILD_TYPE=Release',
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
    
    return pybind_lib


def get_extensions():
    if BUILD_DOCS:
        return []
    
    build_with_cmake()
    
    return []


def get_build_ext():
    return {}


class CustomBuildPy(build_py):
    def run(self):
        build_py.run(self)
        
        pybind_src = os.path.join('python', 'ops_gnn', '_pybind.so')
        if os.path.exists(pybind_src):
            pybind_dest = os.path.join(self.build_lib, 'ops_gnn', '_pybind.so')
            os.makedirs(os.path.dirname(pybind_dest), exist_ok=True)
            shutil.copy2(pybind_src, pybind_dest)
            print(f'Copied _pybind.so to build lib: {pybind_dest}')


class CustomBDistWheel(bdist_wheel):
    def finalize_options(self):
        bdist_wheel.finalize_options(self)
        whl_output_dir = 'output/whl'
        os.makedirs(whl_output_dir, exist_ok=True)
        self.dist_dir = whl_output_dir
    
    def get_tag(self):
        arch = platform.machine()
        platform_tag = f'linux_{arch}'
        return ('py3', 'none', platform_tag)
    
    def run(self):
        bdist_wheel.run(self)
        arch = platform.machine()
        platform_tag = f'linux_{arch}'
        whl_files = glob.glob(os.path.join(self.dist_dir, '*.whl'))
        for whl_file in whl_files:
            base_name = os.path.basename(whl_file)
            new_name = base_name.replace(f'-py3-none-{platform_tag}', f'-{platform_tag}')
            new_path = os.path.join(self.dist_dir, new_name)
            os.rename(whl_file, new_path)


install_requires = []

test_requires = [
    'pytest',
    'pytest-cov',
]

include_package_data = True

def get_cmdclass():
    cmdclass = get_build_ext()
    cmdclass['build_py'] = CustomBuildPy
    cmdclass['bdist_wheel'] = CustomBDistWheel
    return cmdclass


setup(
    name='ops_gnn',
    version=__version__,
    description=(
        'OpsGNN: Library of Optimized Graph Neural Network Algorithms for NPU'
    ),
    author='Ascend',
    author_email='your.email@example.com',
    url=URL,
    download_url=f'{URL}/archive/{__version__}.tar.gz',
    python_requires='>=3.8',
    install_requires=install_requires,
    extras_require={'test': test_requires},
    ext_modules=get_extensions() if not BUILD_DOCS else [],
    cmdclass=get_cmdclass(),
    packages=find_packages('python'),
    package_dir={'': 'python'},
    include_package_data=include_package_data,
)
