import os
from glob import glob
from setuptools import setup, find_packages
from distutils.sysconfig import get_python_inc
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_ext_sources = []
_include_dirs= [os.path.dirname(get_python_inc(plat_specific=1)), ]

root_source = 'src'
for op in os.listdir(root_source):
    # if op == "indice_maxpool":
    #     continue
    dir_op = os.path.join(root_source, op)
    if not os.path.isdir(dir_op):
        continue
    _include_dirs.append(dir_op)
    _ext_sources+=glob(os.path.join(dir_op, "*.cpp"))
    _ext_sources+=glob(os.path.join(dir_op, "*.cu"))
_ext_sources += [os.path.join(root_source, "ss_conv_ops_api.cpp"), os.path.join(root_source, "ss_conv_ops.cu")]

setup(
    name = 'SS-Conv',
    version='1.0.0',
    author='gorilla-lab',
    description='sparse steerable convolution for pytorch',
    packages = find_packages(),
    ext_modules=[
        CUDAExtension(
            name = 'ss_conv._ext', 
            sources = _ext_sources,
            include_dirs = _include_dirs,
            extra_compile_args={'cxx': ['-g', '-std=c++14'], 'nvcc': ['-O2', '-std=c++14']}
        )
    ],
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=True)}
)