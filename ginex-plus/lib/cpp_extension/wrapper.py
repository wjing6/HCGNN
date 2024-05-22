import os
from torch.utils.cpp_extension import load

dir_path = os.path.dirname(os.path.realpath(__file__))

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

sample = load(name='sample', sources=[os.path.join(dir_path, 'sample.cpp')], extra_cflags=['-fopenmp', '-O2'], extra_ldflags=['-lgomp','-lrt'])
# gather = load(name='gather', sources=[os.path.join(dir_path, 'gather.cu')], extra_cuda_cflags=['-Xcompiler', '-fopenmp', '-O2'], extra_ldflags=['-lgomp','-lrt', 'lcudart'], extra_include_paths=[os.path.join(find_cuda(), 'lib64')])
gather = load(name='gather', sources=[os.path.join(dir_path, 'gather.cpp'), ], extra_cflags=['-fopenmp', '-O2'], extra_ldflags=['-lgomp','-lrt', '-laio'], extra_include_paths=['/home/tiger/liuyibo/local/miniconda3/envs/cuda-3.9/include/'])
mt_load = load(name='mt_load', sources=[os.path.join(dir_path, 'mt_load.cpp')], extra_cflags=['-fopenmp', '-O2'], extra_ldflags=['-lgomp','-lrt'])
update = load(name='update', sources=[os.path.join(dir_path, 'update.cpp')], extra_cflags=['-fopenmp', '-O2'], extra_ldflags=['-lgomp','-lrt'])
free = load(name='free', sources=[os.path.join(dir_path, 'free.cpp')], extra_cflags=['-O2'])

