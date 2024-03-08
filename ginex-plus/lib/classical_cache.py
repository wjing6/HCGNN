from collections import OrderedDict
import numpy as np
import torch


class s3FIFO:
    # quick demotion and lazy promotion
    # refer to juncheng-yang's paper
    def __init__(self, cache_entries, pbar, fifo_size_ratio=0.1, fifo_length=500000, threshold=1, use_ratio=False):

        self.num_entries = cache_entries
        self.pbar = pbar
        self.threshold = threshold
        if use_ratio:
            self.fifo_cache_size = int(fifo_size_ratio * cache_entries)
        else:
            self.fifo_cache_size = fifo_length
        self.main_cache_size = self.num_entries - self.fifo_cache_size
        self.ghost_cache_size = self.main_cache_size
        self.fifo_cache = OrderedDict()
        self.main_cache = OrderedDict()
        self.ghost_cache = OrderedDict()  # actually need less memory
        print("fifo length: {:d}, main length: {:d}\n".format(
            self.fifo_cache_size, self.main_cache_size))

    def get(self, key):
        cache_hit = False
        if key in self.fifo_cache:
            self.fifo_cache[key] = min(self.fifo_cache[key] + 1, 10)
            cache_hit = True
        elif key in self.main_cache:
            self.main_cache[key] = min(self.main_cache[key] + 1, 10)
            cache_hit = True
        else:
            self.insert(key, 0)
        return cache_hit

    def insert(self, key, value):
        if key in self.ghost_cache:
            # insert key to main_cache
            if len(self.main_cache) == self.main_cache_size:
                self.evictM()
                self.main_cache[key] = value
            else:
                self.main_cache[key] = value
        else:
            if len(self.fifo_cache) == self.fifo_cache_size:
                self.evictS()
                self.fifo_cache[key] = value
            else:
                self.fifo_cache[key] = value

    def evictS(self):
        evicted = False
        while not evicted and len(self.fifo_cache) > 0:
            item = self.fifo_cache.popitem(last=False)
            if item[1] > self.threshold:  # insert to main_cache
                if len(self.main_cache) == self.main_cache_size:
                    self.evictM()
                    self.main_cache[item[0]] = item[1]
                else:
                    self.main_cache[item[0]] = item[1]
            else:
                if item[0] not in self.ghost_cache:
                    if len(self.ghost_cache) == self.ghost_cache_size:
                        self.ghost_cache.popitem(last=False)
                    self.ghost_cache[item[0]] = item[1]
                evicted = True

    def evictM(self):
        evicted = False
        while not evicted and len(self.main_cache) > 0:
            item = self.main_cache.popitem(last=False)
            if item[1] > 0:
                self.main_cache[item[0]] = item[1] - 1  # re-insert
            else:
                evicted = True


class FIFO:
    def __init__(self, cache_entries, tag, feature_dim, fifo_ratio=0.1, only_indice=True, device='cpu'):
        self.num_entries = cache_entries
        self.tag = tag
        self.fifo_ratio = fifo_ratio
        self.cache_entry_status = torch.full([self.num_entries], -1, dtype=torch.int64, device=self.device)
        self.cache_size = int(fifo_ratio * cache_entries)
        self.device = device
        self.cache = []  # the cached idx
        self.only_indice = only_indice
        if not only_indice:
            self.cache_data = torch.zeros(
                self.cache_size, feature_dim, dtype=torch.float32, device=self.device)
        print("In layer {tag}, the cache entry number is {:d}".format(
            self.cache_size, tag=self.tag))


    def get_hit(self, target_nodes):
        if (target_nodes.device != self.device):
            target_nodes = target_nodes.to(self.device)
        target_nodes_status = self.cache_entry_status[target_nodes]
        cache_hit = target_nodes_status >= 0
        hit_nodes = target_nodes[cache_hit]
        return hit_nodes.shape[0]

    def evit_and_place_indice(self, target_nodes):
        # only used for sampling, not included the updating of feature !
        # now only support CPU, so target_nodes is in CPU
        # TODO: Add support for GPU
        target_nodes_status = self.cache_entry_status[target_nodes]
        cache_no_hit = target_nodes_status == -1
        no_hit_nodes = target_nodes[cache_no_hit]
        pop_num = no_hit_nodes.shape[0] + len(self.cache) - self.cache_size
        if pop_num > 0:
            evit_item = self.cache[0:pop_num]
            self.cache = self.cache[pop_num:]
        
        nodes_place = target_nodes[cache_no_hit]
        self.cache.extend(nodes_place.tolist())
        if pop_num > 0:
            self.cache_entry_status[evit_item] = -1
        self.cache_entry_status[nodes_place] = 1
        # 这里因为不涉及真实结果的获取, 因此无需保留真实idx

    def evit_and_place(self, target_nodes, target_feature):
        # batch evit and update!
        if self.only_indice:
            raise ValueError(
                "only_indice is True, you should not update the feature data. Please use evit_and_place_indice")
        assert (target_nodes.shape[0] == target_feature.shape[0])
        print (f"target shape: {target_nodes.shape}")
        if (target_nodes.shape[0] == 0):
            return
        if (target_nodes.device != self.device):
            target_nodes = target_nodes.to(self.device)
            target_feature = target_feature.to(self.device)
        target_nodes_status = self.cache_entry_status[target_nodes]
        cache_no_hit = target_nodes_status == -1
        no_hit_nodes = target_nodes[cache_no_hit]
        # 由于在调用时, target_nodes应该都不在缓存中(否则在之前sample时应该被剪枝), 因此应该有 len(target_nodes) == cache_no_hit
        assert(target_nodes.shape[0] == no_hit_nodes.shape[0])
        pop_num = no_hit_nodes.shape[0] + len(self.cache) - self.cache_size
        print (f"pop number: {pop_num}")
        if pop_num > 0:
            evit_item = self.cache[0:pop_num]
            print (f"before pop, cache len: {len(self.cache)}")
            self.cache = self.cache[pop_num:]
            # 更新 embedding idx
            self.cache_entry_status[self.cache] -= pop_num
            self.cache_data = self.cache_data[pop_num:, :]
            if pop_num < target_nodes.shape[0]:
                self.cache_data = self.cache_data[0: pop_num - target_nodes.shape[0], :]
            push_idx = torch.tensor(range(len(self.cache), self.cache_size), device=self.device)
            self.cache_data = torch.cat((self.cache_data, target_feature), dim = 0)
        else:
            push_idx = torch.tensor(range(len(self.cache), len(self.cache) + no_hit_nodes.shape[0]), device=self.device)
            self.cache_data[push_idx] = target_feature
        if no_hit_nodes.is_cuda:
            # cache is [], stored in 'CPU'
            self.cache.extend(no_hit_nodes.cpu().tolist())
        print(f"after cat, cache shape: {self.cache_data.shape}")
        if pop_num > 0:
            self.cache_entry_status[evit_item] = -1
        assert(no_hit_nodes.shape[0] == push_idx.shape[0])
        self.cache_entry_status[no_hit_nodes] = push_idx

    def get_hit_nodes(self, target_nodes):
        # return the node that in the embedding cache(will be used as the stale representation)
        if (target_nodes.device != self.device):
            target_nodes = target_nodes.to(self.device)
        target_nodes_status = self.cache_entry_status[target_nodes]
        cache_hit = target_nodes_status >= 0
        target_node_idx = torch.tensor(range(len(target_nodes)), device=self.device)
        hit_nodes_idx = target_node_idx[cache_hit]
        # hit_nodes = target_nodes[cache_hit]
        no_hit_nodes = target_nodes[~cache_hit]
        no_hit_nodes_idx = target_node_idx[~cache_hit]
        pull_embeddings = self.cache_data[target_nodes_status[cache_hit], :]
        return hit_nodes_idx, pull_embeddings, no_hit_nodes_idx, no_hit_nodes

    def get_pop_idx(self, target_nodes):
        # for FIFO, get_pop_idx simply computes the pop_num and return range(pop_num)
        if (target_nodes.device != self.device):
            target_nodes = target_nodes.to(self.device)
        target_nodes_status = self.cache_entry_status[target_nodes]
        cache_no_hit = target_nodes_status == -1
        no_hit_nodes = target_nodes[cache_no_hit]
        pop_num = no_hit_nodes.shape[0] + len(self.cache) - self.cache_size
        return range(pop_num)
    
    def reset(self):
        self.cache_entry_status = torch.full([self.num_entries], -1, dtype=torch.int64, device=self.device)
        self.cache = []  # the cached idx
        print(f"Reset the cache..")
    
    


class LRUCache(OrderedDict):
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value + 1
        else:
            value = None
        return value

    def set(self, key, value):
        item = None
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value + 1
        else:
            if len(self.cache) == self.capacity:
                item = self.cache.popitem(last=False)
                self.cache[key] = value
            else:
                self.cache[key] = value
        return item

    def is_full(self):
        return (len(self.cache) == self.capacity)
