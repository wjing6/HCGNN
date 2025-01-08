from collections import OrderedDict
import numpy as np
import torch
import sys
from log import log
import time
# implementation of s3FIFO and FIFO, now use FIFO for embedding cache

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
        log.info("fifo length: {:d}, main length: {:d}\n".format(
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
    def __init__(self, cache_entries, tag, feature_dim, staleness_thre, fifo_ratio=0.1, only_indice=True, device='cpu'):
        self.num_entries = cache_entries
        self.tag = tag
        self.fifo_ratio = fifo_ratio
        self.device = device
        self.cache_entry_status = torch.full([self.num_entries], -1, dtype=torch.int64, device=self.device)
        self.hit_num = 0
        self.total_place_num = 0
        self.cache_fresh_time = 0
        self.staleness_thre = staleness_thre
        if self.fifo_ratio > 0:
            self.cache_size = int(fifo_ratio * cache_entries)
            self.cache_idx = torch.zeros(self.cache_size, dtype=torch.int64, device=self.device) 
            # the cached idx
            self.only_indice = only_indice

            self.head = 0
            self.cur_len = 0
            if not only_indice:
                self.cache_data = torch.zeros(
                    self.cache_size, feature_dim, dtype=torch.float32, device=self.device)
            log.info("In {tag}, the cache entry number is {:d}".format(
                self.cache_size, tag=self.tag))
    
    def get_hit(self, target_nodes):
        if (target_nodes.device != self.device):
            target_nodes = target_nodes.to(self.device)
        target_nodes_status = self.cache_entry_status[target_nodes]
        cache_hit = target_nodes_status >= 0
        hit_nodes = target_nodes[cache_hit]
        return hit_nodes.shape[0]
    
    def check_if_fresh(self, global_batch):
        if global_batch % self.staleness_thre == 0:
            self.reset()

    def evit_and_place_indice(self, target_nodes, global_batch):
        if self.fifo_ratio == 0:
            return
        # only used for sampling, not included the updating of feature!
        if global_batch % self.staleness_thre == 0:
            self.reset()
            # log.info("receive the threshold, refresh the cache..")
        target_nodes_status = self.cache_entry_status[target_nodes]
        cache_no_hit = target_nodes_status == -1
        no_hit_nodes = target_nodes[cache_no_hit]
        self.hit_num += target_nodes.shape[0] - no_hit_nodes.shape[0]
        self.total_place_num += target_nodes.shape[0]
        pop_num = no_hit_nodes.shape[0] + self.cur_len - self.cache_size
        push_num = no_hit_nodes.shape[0]
        tail = (self.head + self.cur_len) % (self.cache_size)

        if pop_num > 0:
            if self.head + pop_num >= self.cache_size:
                # log.debug("head out of range..")
                evit_item = self.cache_idx[self.head:self.cache_size]
                push_idx = torch.arange(tail, self.cache_size, device=self.device)
                if self.head + pop_num - self.cache_size > 0:
                    push_idx_part2 = torch.arange(0, self.head + pop_num - self.cache_size, device = self.device)
                    push_idx = torch.cat((push_idx, push_idx_part2), dim = 0)
                    evit_item = torch.cat((evit_item, self.cache_idx[0:self.head + pop_num - self.cache_size]), dim = 0)
                self.head = self.head + pop_num - self.cache_size
            else:
                # log.debug(f"{global_batch}, head remain, len: {self.cur_len}, head: {self.head}, tail: {tail}")
                evit_item = self.cache_idx[self.head:self.head + pop_num]
                if (tail <= self.head):
                    # tail = head, 但此时pop > 0, 属于缓存满状态
                    push_idx = torch.arange(tail, self.head + pop_num, device=self.device)
                else:
                    push_idx = torch.arange(tail, self.cache_size, device=self.device)
                    push_idx_part = torch.arange(0, self.head + pop_num, device= self.device)
                    push_idx = torch.cat((push_idx, push_idx_part), dim = 0)
                self.head += pop_num
        else:
            if tail <= self.head or tail + push_num < self.cache_size:
                push_idx = torch.arange(tail, tail + push_num, device=self.device)
            else:
                push_idx = torch.arange(tail, self.cache_size, device=self.device)
                push_idx_part = torch.arange(0, tail + push_num - self.cache_size, device=self.device)
                push_idx = torch.cat((push_idx, push_idx_part), dim = 0)
        # log.debug(f"pop_num: {pop_num}, push_idx.shape {push_idx.shape}, no hit nodes shape: {no_hit_nodes.shape}")
        self.cache_idx[push_idx] = no_hit_nodes
        if pop_num <= 0:
            self.cur_len += push_num
        else:
            self.cur_len = self.cache_size
        if pop_num > 0:
            self.cache_entry_status[evit_item] = -1
        self.cache_entry_status[no_hit_nodes] = 1
        assert self.cur_len <= self.cache_size
        # 这里因为不涉及真实结果的获取, 因此无需保留真实位置

    def evit_and_place(self, target_nodes, target_feature, global_batch):
        if self.fifo_ratio == 0:
            return
        # batch evit and update!
        if self.only_indice:
            log.error(
                "only_indice is True, you should not update the feature data. Please use evit_and_place_indice")
            return
        if global_batch % self.staleness_thre == 0:
            self.reset()
        assert (target_nodes.shape[0] == target_feature.shape[0]), f"current nodes shape:{target_nodes.shape}, feature shape:{target_feature.shape}"
        if (target_nodes.shape[0] == 0):
            log.info("all nodes hit.")
            return
        if (target_nodes.device != self.device):
            target_nodes = target_nodes.to(self.device)
            target_feature = target_feature.to(self.device)
        target_nodes_status = self.cache_entry_status[target_nodes]
        cache_no_hit = target_nodes_status == -1
        no_hit_nodes = target_nodes[cache_no_hit]
        # 由于在调用时, target_nodes应该都不在缓存中(否则在之前sample时应该被剪枝), 因此应该有 len(target_nodes) == cache_no_hit
        assert(target_nodes.shape[0] == no_hit_nodes.shape[0])
        pop_num = no_hit_nodes.shape[0] + self.cur_len - self.cache_size
        push_num = no_hit_nodes.shape[0]
        tail = (self.head + self.cur_len) % (self.cache_size)
        if pop_num > 0:
            # self.cache_entry_status[self.cache] -= pop_num
            # self.cache_data = self.cache_data[pop_num:, :]
            # push_idx = torch.arange(len(self.cache), self.cache_size, device=self.device)
            # self.cache_data = torch.cat((self.cache_data, target_feature), dim = 0)
            # 移动指针，避免移动数据开销
            if self.head + pop_num >= self.cache_size:
                evit_item = self.cache_idx[self.head:self.cache_size]
                push_idx = torch.arange(tail, self.cache_size, device=self.device)
                if self.head + pop_num - self.cache_size > 0:
                    push_idx_part2 = torch.arange(0, self.head + pop_num - self.cache_size, device = self.device)
                    push_idx = torch.cat((push_idx, push_idx_part2), dim = 0)
                    evit_item = torch.cat((evit_item, self.cache_idx[0:self.head + pop_num - self.cache_size]), dim = 0)
                self.cache_data[push_idx] = target_feature
                self.head = self.head + pop_num - self.cache_size
            else:
                evit_item = self.cache_idx[self.head:self.head + pop_num]
                if (tail <= self.head):
                    push_idx = torch.arange(tail, self.head + pop_num, device=self.device)
                else:
                    if (tail + push_num - 1 < self.cache_size):
                        push_idx = torch.arange(tail, tail + push_num, device=self.device)
                    else:
                        push_idx = torch.arange(tail, self.cache_size, device=self.device)
                        push_idx_part = torch.arange(0, self.head + pop_num, device=self.device)
                        push_idx = torch.cat((push_idx, push_idx_part), dim = 0)
                self.cache_data[push_idx] = target_feature
                self.head += pop_num
        else:
            if tail <= self.head or tail + push_num < self.cache_size:
                push_idx = torch.arange(tail, tail + push_num, device=self.device)
            else:
                push_idx = torch.arange(tail, self.cache_size, device=self.device)
                push_idx_part = torch.arange(0, tail + push_num - self.cache_size, device=self.device)
                push_idx = torch.cat((push_idx, push_idx_part), dim = 0)
            self.cache_data[push_idx] = target_feature

        self.cache_idx[push_idx] = no_hit_nodes
        
        if pop_num > 0:
            self.cache_entry_status[evit_item] = -1
            self.cur_len = self.cache_size
        else:
            self.cur_len = self.cur_len + push_num
        assert(no_hit_nodes.shape[0] == push_idx.shape[0]), f"current push_idx shape: {push_idx.shape}, nodes shape: {no_hit_nodes.shape}"
        self.cache_entry_status[no_hit_nodes] = push_idx

    def get_hit_nodes(self, target_nodes, global_batch):
        # return the node that in the embedding cache(will be used as the stale representation)
        # need to check the staleness.. if stale, return None
        if global_batch % self.staleness_thre == 0:
            # log.info(f"{global_batch}, hit None(because refresh)")
            return None, None, None, target_nodes
        if (target_nodes.device != self.device):
            target_nodes = target_nodes.to(self.device)
        target_nodes_status = self.cache_entry_status[target_nodes]
        cache_hit = target_nodes_status >= 0
        target_node_idx = torch.arange(0, len(target_nodes), device=self.device)
        hit_nodes_idx = target_node_idx[cache_hit]
        # hit_nodes = target_nodes[cache_hit]
        no_hit_nodes = target_nodes[~cache_hit]
        no_hit_nodes_idx = target_node_idx[~cache_hit]
        pull_embeddings = self.cache_data[target_nodes_status[cache_hit], :]
        return hit_nodes_idx, pull_embeddings, no_hit_nodes_idx, no_hit_nodes

    def get_pop_idx(self, target_nodes):
        # for FIFO, get_pop_idx simply computes the pop_num and return range(pop_num)
        # FIXME: currently unused!
        if (target_nodes.device != self.device):
            target_nodes = target_nodes.to(self.device)
        target_nodes_status = self.cache_entry_status[target_nodes]
        cache_no_hit = target_nodes_status == -1
        no_hit_nodes = target_nodes[cache_no_hit]
        pop_num = no_hit_nodes.shape[0] + len(self.cache) - self.cache_size
        return range(pop_num)
    
    def reset(self):
        # 不回收实际缓存
        if self.fifo_ratio == 0:
            return 
        start = time.time()
        self.cache_entry_status = self.cache_entry_status.fill_(-1)
        self.head = 0
        self.cur_len = 0
        self.cache_fresh_time += time.time() - start

    
    


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
