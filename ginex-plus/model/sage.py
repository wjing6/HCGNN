import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.nn import SAGEConv
from typing import List
from lib.classical_cache import FIFO
import numpy as np

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, embedding_size: List[int]):
        super(SAGE, self).__init__()

        if len(embedding_size) != num_layers - 1:
            raise ValueError(f"embedding size should be {num_layers - 1} but got {len(embedding_size)}")

        self.num_layers = num_layers
        self.embedding_cache = {}
        for layer in range(len(embedding_size)):
            tmp_tag = 'layer_' + str(layer + 1)
            self.embedding_cache[tmp_tag] = FIFO(embedding_size[layer], tmp_tag, hidden_channels, 1, only_indice=False)
            # 1 表示 size 值已经在参数'1'给出

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))


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
            print (f"x_target shape: {x_target.shape}, x shape: {x.shape}")
            x_target_nid = n_id[x_target]
            if i < self.num_layers - 1:
                pull_nodes_idx, pull_embeddings = self.push_and_pull(x, x_target_nid, self.num_layers - i - 1)
                x[pull_nodes_idx] = pull_embeddings
                print (f"after pull, x shape: {x.shape}")
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def push_and_pull(self, full_embeddings, x_target, layer):
        # push the updating embedding into the cache and pull the stale embedding to the corresponding 'tensor'
        # 'x_target' include all the nodes, we need to fetch the embedding with the idx in 'x_target'
        # 'push_embedding' corresponds with 'push_idx'
        pull_node_idx, pull_nodes, push_node_idx, push_nodes = self.embedding_cache[layer].get_hit_nodes(x_target)
        # pull_node_idx 对应 cache 中的 idx, 还需要根据 pull_nodes 得到对应 x_target 位置
        push_embeddings = full_embeddings.index_select(0, push_node_idx)
        pull_node_embeddings = torch.tensor(self.embedding_cache[layer]).index_select(0, pull_node_idx)
        self.embedding_cache[layer].evit_and_place(push_nodes, push_embeddings)
        pull_node_idx_in_target = []
        # TODO: naive implementation, need improve!
        for node in pull_nodes:
            pull_node_idx_in_target.append(np.where(x_target == node)[0].squeeze())
        pull_node_idx_in_target = torch.tensor(pull_node_idx_in_target)
        return pull_node_idx_in_target, pull_node_embeddings


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