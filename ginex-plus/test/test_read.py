import torch
import test
import os
import sys
import numpy as np
from random import sample 
from ogb.nodeproppred import PygNodePropPredDataset
import time

num_thread = os.cpu_count()
os.environ['TEST_NUM_THREADS'] = str(num_thread)

dataset = 'ogbn-papers100M'
path = '/data01/liuyibo/papers'
features_path = os.path.join(path, 'features.dat')
features_shape = [111059956, 128]
# features = np.memmap(features_path, mode='r', shape=tuple(features_shape), dtype=np.float32)
# features = torch.from_numpy(features)

dataset = PygNodePropPredDataset('ogbn-papers100M', path)

gather_idx = sample(range(111059956), 200000)
print ("=" * 20 + "random generating finished" + "=" * 20)

gather_idx = torch.Tensor(gather_idx).long()
start = time.time()
test.test_direct_copy(features_path, gather_idx, 111059956, 128)
print ("cost: {:.4f} s".format(time.time() - start))