import torch
from os import path
import numpy as np
import math
import time
from typing import List
from ..utils import log

class HistoryCache:

    def __init__(self, num_nodes, size,
    hidden_dimension, method, staleness_thres, device, layer_id):
        # TODO: using Config, reduce the args size
        self.hidden_dimension = hidden_dimension
        self.size = size
        self.num_nodes = num_nodes
        self.device = device
        self.method = method
        self.layer_id = layer_id
        self.head = 0 # head 指明下一步evit的数据
        # history embedding settings
        assert self.method in ["random", "grad", "staleness", "fifo"]
        assert self.num_layers > 0
        self.staleness_thres = int(staleness_thres)
        
        self.num2index = torch.ones(self.num_nodes, dtype=torch.int64,
                                    device=self.device) * -1

        self.produced_embedding = None  # embedding that produced from training iterations

    def allocate(self):
        t0 = time.time()
        max_byte = torch.cuda.mem_get_info()[0]
        # reverve 8GB memory for other usage
        max_byte = max_byte - 8 * 1024 * 1024 * 1024
        max_buffer_byte = max_byte
        if self.limit > 0 and max_buffer_byte > self.size:
            max_buffer_byte = int(self.size)
        max_buffer_byte = (max_buffer_byte + 4 * self.hidden_dimension - 1) // (4 * self.hidden_dimension) \
                            * (4 * self.hidden_dimension)
        self.num_cache_embedding = max_buffer_byte // 4
        self.buffer = torch.empty(self.num_cache_embedding, device=self.device)
        # 4 byte = sizeof(float)
        log.info(f"max buffer size: {self.buffer.shape}")
        self.max_buffer_byte = max_buffer_byte
        log.info(f"allocating buffer: {time.time() - t0}")

    @torch.no_grad()  # important for removing grad from computation graph
    def update_history(self, batch, glb_iter):
        if self.staleness_thres == 0:
            return
        if glb_iter < self.start_history:
            # not starting historical embedding
            return
        num_node = batch.size[0]
        

        if (glb_iter % self.staleness_thres) == 0:
            self.num_stored_history = max(self.num_stored_history, self.header)
            self.header = 0

        if self.produced_embedding is None:
            log.info("produced_embedding is None, maybe some error in invoking.. ")
            self.produced_embedding = torch.randn(
                [num_node, self.hidden_channels], device=self.device)
        if self.produced_embedding.grad is not None and self.method == "grad":
            # Gradient policy
            grad = self.produced_embedding.grad
            assert grad.shape[0] == num_node, "grad shape: {}, num_node: {}".format(
                    grad.shape, num_node)
            grad = grad.norm(dim=1)
        else:
            # Random policy
            grad = torch.randn([num_node], device=self.device)
        self.produced_embedding.grad = None

        # TODO: how to find an appropriate one - self.rate?
        thres = torch.quantile(grad, self.rate)

        # Record
        # 1. grad less than thres; 2. not the cached embeddings
        record_mask = torch.logical_and(grad < thres, self.subStatus == -1)
        embed_to_place = self.produced_embedding[record_mask]
        num_record = embed_to_place.shape[0]
        if self.header + num_record < self.size:
            self.buffer.view(-1, self.hidden_dimension)[self.header:self.header + num_record] = embed_to_place
        # invalidate the previous cached **features**
        begin = self.header * self.hidden_channels // self.in_channels
        end = math.ceil((self.header + num_to_record) * self.hidden_channels /
                        self.in_channels)
        self.full2embed[self.feat2full[begin:end]] = -1

        # invalidate the cache that are about to be overwritten
        # NOTICE: embed2full[i] = j does not mean full2embed[j] = i
        invalid_fulls = self.embed2full[self.header:self.header +
                                        num_to_record]
        change_area = self.full2embed[invalid_fulls]
        # Only when full2embed is pointing to [header, header + num_to_record), we invalidate it
        # invalidated_id = invalid_fulls[torch.logical_and(
        #     change_area >= self.header,
        #     change_area < self.header + num_to_record)]
        change_area[torch.logical_and(
            change_area >= self.header,
            change_area < self.header + num_to_record)] = -1
        self.full2embed[invalid_fulls] = change_area

        # Update the mapping from full id to embed id
        change_id = batch.sub_to_full[:num_node][record_mask]
        # the follow assert is correct
        # assert (torch.sum(self.full2embed[change_id] != -1) == 0)
        new_id = torch.arange(self.header,
                              self.header + num_to_record,
                              device=self.device)
        # Map between full ID and embedding ID
        self.full2embed[change_id] = new_id
        self.embed2full[self.header:self.header +
                        num_to_record] = batch.sub_to_full[:num_node][
                            record_mask]
        self.header += num_to_record
        self.produced_embedding = None

    # infer both node features and history embeddings
    def lookup_and_load(self, batch, num_layer, feature_cache_only=False):
        if feature_cache_only:  # only use feature cache, do not use embedding cache
            self.sub2feat = self.full2embed[batch.sub_to_full +
                                            self.total_num_node]
            load_mask = self.sub2feat == -1
            if self.distributed_store:  # load from other GPUs
                assert False
            else:  # load from UVM
                batch.x = self.uvm.masked_get(batch.sub_to_full, load_mask)
            cached_feat = self.buffer.view(
                -1, self.in_channels)[self.sub2feat]  # load from feat cache
            cached_feat[load_mask] = 0
            batch.x += cached_feat  # combine feat from UVM and feat cache
            return None

        if not "history" in self.feat_mode:  # no cache
            self.sub2embed = torch.ones([batch.num_node_in_layer[1].item()],
                                        dtype=torch.int64,
                                        device=self.device) * -1
            self.cached_embedding = torch.tensor([])
            if self.distributed_store:
                return torch.ones(batch.sub_to_full.shape[0],
                                  device=self.device,
                                  dtype=torch.bool)
            else:
                if self.feat_mode in ["uvm", "mmap"]:
                    batch.x = self.uvm.get(batch.sub_to_full)
                else:
                    assert False
                return None

        # nodes that are connected to seed nodes
        layer1_nodes = batch.sub_to_full[:batch.num_node_in_layer[1].item()]
        input_nodes = batch.sub_to_full
        # whether nodes have cached embeddings
        self.sub2embed = self.full2embed[layer1_nodes]
        # 1. load cached embedding
        self.cached_embedding = self.buffer.view(
            -1, self.hidden_channels)[self.sub2embed]
        # whether input nodes have cached features
        self.sub2feat = self.full2embed[input_nodes + self.total_num_node]
        # torch.cuda.synchronize()
        self.used_masks = hiscache.count_history_reconstruct(
            batch.ptr,
            batch.idx,
            self.sub2embed,
            batch.sub_to_full.shape[0],  # num_node
            batch.y.shape[0],  # num_label
            num_layer,  # num_layer
        )
        # torch.cuda.synchronize()
        # input nodes that are not pruned by embeddings
        used_mask = self.used_masks[0]
        hit_feat_mask = self.sub2feat != -1
        # not pruned and not cached
        load_mask = torch.logical_and(used_mask, (~hit_feat_mask))
        self.ana = False
        if self.ana:
            log.info(
                f"glb-iter: {self.glb_iter} prune-by-his: {torch.sum(~self.used_masks[0])} prune-by-feat: {torch.sum(hit_feat_mask)} prune-by-both: {torch.sum(~load_mask)} overall: {batch.sub_to_full.shape[0]} hit-rate {torch.sum(~load_mask) / batch.sub_to_full.shape[0]}"
            )
            embed_num = torch.sum(self.full2embed[:self.total_num_node] != -1)
            feat_num = torch.sum(self.full2embed[self.total_num_node:] != -1)
            overall_size = feat_num * self.in_channels + embed_num * self.hidden_channels
            log.info(
                f"embed-num: {embed_num} feat-num: {feat_num} overall: {overall_size} buffersize: {self.buffer.shape[0]}"
            )
        self.glb_iter += 1
        if not self.distributed_store:
            # 2. load raw features
            x = self.uvm.masked_get(batch.sub_to_full, load_mask)
            # 3. load hit raw feature cache
            cached_feat = self.buffer.view(-1, self.in_channels)[self.sub2feat]
            cached_feat[~hit_feat_mask] = 0
            x += cached_feat
            batch.x = x
            return None
        else:
            cached_feat = self.buffer.view(-1, self.in_channels)[self.sub2feat]
            cached_feat[~hit_feat_mask] = 0
            batch.x = cached_feat
            return load_mask  # return the mask of feature that needs communication