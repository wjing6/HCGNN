from collections import OrderedDict
import numpy as np
import torch

class s3FIFO:
    # quick demotion and lazy promotion
    # refer to juncheng-yang's paper
    def __init__(self, cache_entries, pbar, fifo_size_ratio=0.1, fifo_length=500000, threshold=1, use_ratio = False):

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
        self.ghost_cache = OrderedDict() # actually need less memory
        print ("fifo length: {:d}, main length: {:d}\n".format(self.fifo_cache_size, self.main_cache_size))
    
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
            if item[1] > self.threshold: # insert to main_cache
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
                self.main_cache[item[0]] = item[1] - 1 # re-insert
            else:
                evicted = True


class FIFO:
    def __init__(self, cache_entries, tag, cache_entry_status, feature_dim, fifo_ratio=0.1):
        self.num_entries = cache_entries
        self.tag = tag
        self.fifo_ratio = fifo_ratio
        self.cache_entry_status = cache_entry_status
        self.cache_size = int(fifo_ratio * cache_entries)
        self.cache = [] # the cached idx
        self.cache_data = torch.zeros(self.cache_size, feature_dim, dtype=torch.float32)
        print("In layer {tag}, the cache entry number is {:d}".format(
            self.num_entries, tag=self.tag))

    def get(self, key):
        if key in self.cache:
            return True
        else:
            if len(self.cache) == self.cache_size:
                evit_item = self.cache.pop(0)
                self.cache_entry_status[evit_item] = 0
                self.add(key)
            else:
                self.add(key)
            return False

    def add(self, key):
        if len(self.cache) == self.cache_size:
            print("try to put an item to a full list, abort")
            return
        self.cache.append(key)
        self.cache_entry_status[key] = 1

    def print_debug(self):
        res = "curr cache item: "
        for item in self.cache:
            res += str(item)
            res += " ,"
        print(res)
    
    def get_hit(self, target_nodes):
        target_nodes_status = self.cache_entry_status[target_nodes]
        cache_hit = target_nodes_status > 0
        target_nodes = np.array(target_nodes)
        hit_nodes = target_nodes[cache_hit]
        return len(hit_nodes)
    
    def evit_and_place_indice(self, target_nodes):
        # only used for sampling, not included the updating of feature !
        target_nodes_status = self.cache_entry_status[target_nodes]
        cache_no_hit = target_nodes_status == 0
        pop_num = len(cache_no_hit) + len(self.cache) - self.cache_size
        if pop_num > 0:
            evit_item = self.cache[0:pop_num]
            self.cache = self.cache[pop_num:]
        target_nodes = np.array(target_nodes)
        nodes_place = target_nodes[cache_no_hit]
        self.cache.extend(nodes_place)
        if pop_num > 0:
            self.cache_entry_status[evit_item] = 0
        self.cache_entry_status[nodes_place] = 1

    def evit_and_place(self, target_nodes, target_feature):
        # batch evit and update!
        assert(target_nodes == target_feature.shape[0])

        target_nodes_status = self.cache_entry_status[target_nodes]
        cache_no_hit = target_nodes_status == 0
        pop_num = len(cache_no_hit) + len(self.cache) - self.cache_size
        if pop_num > 0:
            evit_item = self.cache[0:pop_num]
            self.cache = self.cache[pop_num:]
            self.cache_data = self.cache_data[pop_num:,:]
            print (f"layer: {self.tag}, evit.. pop number: {pop_num}, after pop, the cache shape: {self.cache_data.shape}")
        target_nodes = np.array(target_nodes)
        nodes_place = target_nodes[cache_no_hit]
        self.cache.extend(nodes_place)
        self.cache_data = torch.cat(self.cache_data, target_feature, dim=0)
        print (f"after cat, cache shape: {self.cache_data.shape}")
        if pop_num > 0:
            self.cache_entry_status[evit_item] = 0
        self.cache_entry_status[nodes_place] = 1

    @property
    def get_status(self):
        return self.cache_entry_status

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
