import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.nn import SAGEConv
from typing import List
from lib.classical_cache import FIFO
import numpy as np
import time

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_nodes, 
                device, cache_device, stale_thre, embedding_rate: List[float]):
        super(SAGE, self).__init__()

        if len(embedding_rate) != num_layers - 1:
            raise ValueError(f"embedding size should be {num_layers - 1} but got {len(embedding_rate)}")

        self.num_layers = num_layers
        self.embedding_cache = {}
        self.device = device
        # model device
        self.cache_device = cache_device
        # default: cache device = model device, in the same GPU
        for layer in range(len(embedding_rate)):
            tmp_tag = 'layer_' + str(layer + 1)
            self.embedding_cache[tmp_tag] = FIFO(num_nodes, tmp_tag, hidden_channels, stale_thre, embedding_rate[layer], 
            device=self.cache_device, only_indice=False)


        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        
        self.cur_batch = 0
        # timer
        self.index_select_time = 0
        self.evit_time = 0
        self.cache_transfer_time = 0


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()


    def forward(self, x, adjs, n_id):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            # 'nid' need to get the global id of target
            x = self.convs[i]((x, x_target), edge_index)
            # e.g. x_target: [1, 2, 3, 5, 8], in embedding_cache: [3, 5]
            # which means there is no edge_index with 3 and 5
            # 根据 convs 定义, 实际得到的 x 维度与不裁剪应该是一致的
            x_target_nid = n_id[:size[1]]
            if i < self.num_layers - 1:
                pull_nodes_idx, pull_embeddings = self.push_and_pull(x, x_target_nid, self.num_layers - i - 1)
                if pull_nodes_idx is not None and pull_nodes_idx.shape[0] != 0:
                    if pull_nodes_idx.device != self.device:
                    # embedding is not the same as training device, transfer the cache to the training device
                        cache_transfer_start = time.time()
                        pull_nodes_idx = pull_nodes_idx.to(self.device)
                        pull_embeddings = pull_embeddings.to(self.device)
                        self.cache_transfer_time += time.time() - cache_transfer_start
                    x[pull_nodes_idx] = pull_embeddings
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        self.cur_batch += 1
        return x.log_softmax(dim=-1)

    def push_and_pull(self, full_embeddings, x_target, layer):
        # push the updating embedding into the cache and pull the stale embedding to the corresponding 'tensor'
        # 'x_target' include all the nodes, we need to fetch the embedding with the idx in 'x_target'
        # 'push_embedding' corresponds with 'push_idx'
        assert(full_embeddings.shape[0] == x_target.shape[0]), f"err in push_and_pull, embedding shape:{full_embeddings.shape} not match node shape:{x_target.shape}"
        if full_embeddings.device != self.cache_device:
            full_embeddings = full_embeddings.to(self.cache_device)
            x_target = x_target.to(self.cache_device)
        layer_tag = 'layer_' + str(layer)
        pull_nodes_idx, pull_embeddings, push_nodes_idx, push_nodes = self.embedding_cache[layer_tag].get_hit_nodes(x_target, self.cur_batch)
        # pull_node_idx 对应 x_target 中的idx
        if pull_nodes_idx is not None:
            index_select_timer = time.time()
            push_embeddings = full_embeddings.index_select(0, push_nodes_idx).clone().detach()
            # remove from the computation graph
            self.index_select_time += time.time() - index_select_timer
            evit_time_start = time.time()
            self.embedding_cache[layer_tag].evit_and_place(push_nodes, push_embeddings, self.cur_batch)
            self.evit_time += time.time() - evit_time_start
        else:
            push_embeddings = full_embeddings.clone().detach()
            self.embedding_cache[layer_tag].evit_and_place(push_nodes, push_embeddings, self.cur_batch)
        return pull_nodes_idx, pull_embeddings

    def reset_embeddings(self):
        # after each epoch, remove the embeddings
        for layer in range(self.num_layers - 1):
            tmp_tag = 'layer_' + str(layer + 1)
            self.embedding_cache[tmp_tag].reset()
        
        self.evit_time = 0
        self.index_select_time = 0
        self.cache_transfer_time = 0
        self.cur_batch = 0
        
    def inference(self, x_all):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())
                pbar.update(batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all
