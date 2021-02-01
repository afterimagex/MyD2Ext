#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.


import torch

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 5], "Requires PyTorch >= 1.5"

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='d2dev',
    version='0.1',
    description='Fast and accurate single shot object detector',
    author='NVIDIA Corporation',
    author_email='fchabert@nvidia.com',
    packages=[],
    ext_modules=[CUDAExtension('d2dev._C',
                               ['extensions.cpp', 'cuda/nms.cu', 'cuda/bbox_decode.cu'],
                               extra_compile_args={
                                   'cxx': ['-std=c++14', '-O2', '-Wall'],
                                   'nvcc': [
                                       '-std=c++14', '--expt-extended-lambda', '--use_fast_math', '-Xcompiler', '-Wall',
                                       '-gencode=arch=compute_60,code=sm_60', '-gencode=arch=compute_61,code=sm_61',
                                       '-gencode=arch=compute_70,code=sm_70', '-gencode=arch=compute_72,code=sm_72',
                                       '-gencode=arch=compute_75,code=sm_75', '-gencode=arch=compute_75,code=compute_75'
                                   ],
                               },
                               libraries=['nvinfer', 'nvinfer_plugin', 'nvonnxparser'],
                               library_dirs=['/usr/local/lib/python3.6/dist-packages/torch/lib'],
                               )
                 ],
    cmdclass={'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)},
    install_requires=[
        'torch>=1.0.0a0',
        'torchvision',
        'pillow==6.2.2',
        'requests',
    ],
    entry_points={'console_scripts': ['d2dev=d2dev.main:main']}
)
