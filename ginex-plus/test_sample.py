import os.path as osp

import torch
import torch.nn.functional as F
from tqdm import tqdm
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborSampler
from random import sample
import quiver
import time
import heapq
from collections import OrderedDict
from lib.belady_cache import beladyCache
import random
from random import sample
from lib.prepare_data import edgeList

UNITS = {
    #
    "KB": 2**10,
    "MB": 2**20,
    "GB": 2**30,
    #
    "K": 2**10,
    "M": 2**20,
    "G": 2**30,
}


def parse_size(sz) -> int:
    if isinstance(sz, int):
        return sz
    elif isinstance(sz, float):
        return int(sz)
    elif isinstance(sz, str):
        for suf, u in sorted(UNITS.items()):
            if sz.upper().endswith(suf):
                return int(float(sz[:-len(suf)]) * u)
    raise Exception("invalid size: {}".format(sz))


reindex = dict()
totalIndex = 0
class LRUCache(OrderedDict):
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()
     

    def get(self,key):
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
        else:
            value = None
        return value
     

    def set(self,key,value):
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
        else:
            if len(self.cache) == self.capacity:
                self.cache.popitem(last = False)
                self.cache[key] = value
            else:
                self.cache[key] = value


root = "./utils/b.edgelist"
# =================
# root = "/data01/liuyibo/products"
# dataset = PygNodePropPredDataset('ogbn-products', root)
# split_idx = dataset.get_idx_split()
# data = dataset[0]
# =================

# num_features = data.x.shape[1]
# num_nodes = data.num_nodes
num_features = 128
feature_byte = num_features * 8
cache_size = parse_size("100MB") // feature_byte
# cache = LRUCache(cache_size)

# print (split_idx['train'].shape, split_idx['test'].shape, split_idx['valid'].shape)


edgeL = edgeList(root)
edge_index, num_nodes = edgeL.getEdgeIndex()
# train_idx = split_idx['train']

train_idx = sample(range(num_nodes), int(0.1 * num_nodes))
csr_topo = quiver.CSRTopo(edge_index) # Quiver
quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, sizes=[20, 10], device=0, mode='GPU') # Quiver



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# x = quiver.Feature(rank=0, device_list=[0], device_cache_size="4G", cache_policy="device_replicate", csr_topo=csr_topo) # Quiver
# x.from_cpu_tensor(data.x) # Quiver

# y = data.y.squeeze().to(device)


def test_sampler(epoch, num_sb, batch_size, sb_size):

    # pbar = tqdm(total=train_idx.size(0))
    pbar = tqdm(total=len(train_idx))
    pbar.set_description(f'Epoch {epoch:02d}')

    ############################################
    # Step 3: Training the PyG Model with Quiver
    ############################################
    # for batch_size, n_id, adjs in train_loader: # Original PyG Code
    start = time.time_ns()
    
    global totalIndex
    for cur_sb in tqdm(range(num_sb)):
        start_idx = cur_sb * batch_size * sb_size
        end_idx = min((cur_sb + 1) * batch_size * sb_size, len(train_idx))
        n_id_list = []
        train_loader = torch.utils.data.DataLoader(train_idx[start_idx : end_idx],
                                           batch_size=batch_size,
                                           shuffle=False) # Quiver
        for iter, seeds in enumerate(train_loader): # Quiver
            n_id, batch_size, adjs = quiver_sampler.sample(seeds) # Quiver
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            # adjs = [adj.to(device) for adj in adjs]
            
            # ========================  LRU cache simulation  ==========================
            # before = len(totalIte)
            # totalIte.update(set(n_id.tolist()))
            # updateNum = len(totalIte) - before
            # for id in n_id.tolist():
            #     cache.set(id, iter)
            # totalIte = set(cache.cache.keys())
            # ==========================================================================
            miss = 0
            for id in n_id.tolist():
                if id not in reindex:
                    reindex[id] = totalIndex
                    totalIndex += 1
                    miss += 1
            # pbar.write("current node: {:d}, miss node: {:d}, total rate: \
            #             {:.4f}%, miss rate: {:.4f}%".format(len(n_id.tolist()), \
            #             miss, float(totalIndex) / num_nodes * 100, float(miss) / len(n_id.tolist()) * 100))
            n_id_list.append(torch.tensor(list(map(lambda x: reindex[x], n_id.tolist()))))
            pbar.update(batch_size)
            # pbar.write("{:.4f}%".format(float(len(n_id.tolist()) - updateNum) / len(n_id.tolist()) * 100))
        belady_cache_sim = beladyCache(len(n_id_list), n_id_list, totalIndex, cache_size)
        belady_cache_sim.simulate()
    floatT = time.time_ns() - start
    pbar.write("floating time:{:.6f} ms".format(floatT / 1e6))
    pbar.close()
    
    print ("total node: {:d}, appearing node number: {:d}".format(num_nodes, totalIndex))

    # =========================== Belady Cache Simulate =====================
        
    # =======================================================================
    return


# pre_index = dict.copy(reindex)
# for epoch in range(1, 5):
#     test_sampler(epoch)
#     print("{:.4f}%".format(float (len(set(pre_index.keys()) & set(reindex.keys()))) / len(set(reindex.keys())) * 100))
#     pre_index = dict.copy(reindex)
#     reindex.clear()

if __name__=='__main__':
    sb_size = 1000
    batch_size = 1000
    num_sb = int((len(train_idx) - 1) / (sb_size * batch_size)) + 1
    test_sampler(1, num_sb, batch_size, sb_size)