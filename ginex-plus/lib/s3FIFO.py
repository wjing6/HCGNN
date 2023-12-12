import torch
from collections import OrderedDict
# using orderedDict, naive implementation

class s3FIFO:
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