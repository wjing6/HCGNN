import glob
import os

from setuptools import find_packages, setup
from torch.utils import cpp_extension


def find_cuda():
    # TODO: find cuda
    home = os.getenv("CUDA_HOME")
    path = os.getenv("CUDA_PATH")
    if home is not None:
        return home
    elif path is not None:
        return path
    else:
        return '/usr/local/cuda'

def find_nccl(): # need identify nccl path
    home = os.getenv("NCCL_HOME")
    return home

def have_cuda():
    if os.getenv('QUIVER_ENABLE_CUDA') == '1': return True
    import torch
    return torch.cuda.is_available()


def create_extension(with_cuda=False):
    print('Building torch_quiver with CUDA:', with_cuda)
    srcs = []
    srcs += glob.glob('./*.cpp')
    srcs += glob.glob('../utils/log/*.cpp')
    include_dirs = [
        os.path.join(os.getcwd(), './'),
        os.path.join(os.getcwd(), '../utils/log/'),
        os.path.join(os.getcwd(), '../utils/')
    ]
    
    library_dirs = []
    libraries = []
    extra_cxx_flags = [
        '-std=c++17',
        '-DUSE_LOG',
        '-fopenmp'
        # TODO: enforce strict build
        # '-Wall',
        # '-Werror',
        # '-Wfatal-errors',
    ]
    if with_cuda:
        cuda_home = find_cuda()
        nccl_home = find_nccl()
        include_dirs += [os.path.join(cuda_home, 'include')]
        include_dirs += [os.path.join(nccl_home, 'include')]
        library_dirs += [os.path.join(cuda_home, 'lib64')]
        library_dirs += [os.path.join(nccl_home, 'lib')]
        # srcs += glob.glob('srcs/cpp/src/quiver/cuda/*.cpp')
        # srcs += glob.glob('srcs/cpp/src/quiver/cuda/*.cu')
        extra_cxx_flags += ['-DHAVE_CUDA']

    print (include_dirs)
    if os.getenv('QUIVER_ENABLE_TRACE'):
        extra_cxx_flags += ['-DQUIVER_ENABLE_TRACE=1']

    return cpp_extension.CppExtension(
        'prepare_data_from_scratch',
        srcs,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        # with_cuda=with_cuda,
        extra_compile_args={
            'cxx': extra_cxx_flags,
            'nvcc': ['-O3', '--expt-extended-lambda', '-Xcompiler', '-fopenmp', '-lnuma', '-lcudart', '-DUSE_LOG'],
        },
    )



setup(
    name='prepare_data_from_scratch',
    license='Apache',
    python_requires='>=3.6',
    ext_modules=[
        create_extension(have_cuda()),
    ],
    cmdclass={
        # FIXME: parallel build, (pip_install took 1m16s)
        'build_ext': cpp_extension.BuildExtension,
    },
)
