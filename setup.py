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
import platform
import shutil
import subprocess

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel

__version__ = '0.1.0'
URL = 'https://gitcode.com/cann/ops-gnn'

BUILD_DOCS = os.getenv('BUILD_DOCS', '0') == '1'

def build_with_cmake():
    if 'ASCEND_HOME_PATH' not in os.environ:
        raise EnvironmentError("ASCEND_HOME_PATH environment variable not set. Please source set_env.sh first.")
    
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

if not BUILD_DOCS:
    build_with_cmake()

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
        platform_tag = f'linux_{platform.machine()}'
        return ('py3', 'none', platform_tag)
    
    def run(self):
        bdist_wheel.run(self)
        platform_tag = f'linux_{platform.machine()}'
        whl_files = glob.glob(os.path.join(self.dist_dir, '*.whl'))
        for whl_file in whl_files:
            base_name = os.path.basename(whl_file)
            new_name = base_name.replace(f'-py3-none-{platform_tag}', f'-{platform_tag}')
            os.rename(whl_file, os.path.join(self.dist_dir, new_name))

setup(
    name='ops_gnn',
    version=__version__,
    description='OpsGNN: Library of Optimized Graph Neural Network Algorithms for NPU',
    author='Ascend',
    url=URL,
    download_url=f'{URL}/archive/{__version__}.tar.gz',
    python_requires='>=3.8',
    extras_require={'test': ['pytest', 'pytest-cov']},
    ext_modules=[],
    cmdclass={
        'build_py': CustomBuildPy,
        'bdist_wheel': CustomBDistWheel,
    },
    packages=find_packages('python'),
    package_dir={'': 'python'},
    include_package_data=True,
)
