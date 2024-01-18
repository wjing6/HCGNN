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

loader = load(name='prepare_data_from_scratch', sources=[os.path.join(dir_path, 'prepare.cpp')], extra_cflags=['-O2'])

