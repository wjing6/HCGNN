# Reaches around 0.7870 ± 0.0036 test accuracy.

import os.path as osp

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv
import time
import argparse
import os
import threading
from queue import Queue
import glob
from datetime import datetime
# from lib.utils import tensor_free

import quiver
from quiver.pyg import GraphSageSampler

import gather_gpu

def prepareMMAPFeature(dataset_path, features):
    os.makedirs(dataset_path, exist_ok=True)
    features_path = os.path.join(dataset_path, 'features.dat')
    print('Saving features...')
    features_mmap = np.memmap(features_path, mode='w+', shape=features.shape, dtype=np.float32)
    features_mmap[:] = features[:]
    features_mmap.flush()
    print('Done!')
    
    features = torch.from_numpy(features_mmap)
    return features, features_path

def loadingMMAPFeature(features_path, shape):
    features = np.memmap(features_path, mode='r', shape=shape, dtype=np.float32)
    features = torch.from_numpy(features)
    return features

KB = 2 ** 10
MB = 2 ** 20
GB = 2 ** 30

sample_time = 0
gather_time = 0
transfer_time = 0
train_time = 0
ssdW_time = 0
read_time = 0

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, default='ogbn-papers100M')
argparser.add_argument('--num-epochs', type=int, default=10)
argparser.add_argument('--batch-size', type=int, default=4096)
argparser.add_argument('--num-workers', type=int, default=os.cpu_count()*2)
argparser.add_argument('--exp-name', type=str, default=None)
argparser.add_argument('--sb-size', type=int, default='100')
argparser.add_argument('--feature-cache-size', type=float, default=500000000)
argparser.add_argument('--trace-load-num-threads', type=int, default=4)
argparser.add_argument('--gather-num-threads', type=int, default=os.cpu_count()*8)
argparser.add_argument('--verbose', dest='verbose', default=False, action='store_true')
argparser.add_argument('--train-only', dest='train_only', default=False, action='store_true')
args = argparser.parse_args()

if args.exp_name is None:
    now = datetime.now()
    args.exp_name = now.strftime('%Y_%m_%d_%H_%M_%S')
os.makedirs(os.path.join('/data01/liuyibo/trace', args.exp_name), exist_ok=True)
root = "/data01/liuyibo/"
dataset = PygNodePropPredDataset(args.dataset, root)
dataset_path = os.path.join(root, args.dataset + '-ginex')

split_idx = dataset.get_idx_split()
evaluator = Evaluator(name=args.dataset)
data = dataset[0]

num_nodes = data.num_nodes
shape = data.x.shape
labels = data.y.squeeze()
train_idx = dataset.get_idx_split()['train']

features_path = os.path.join(dataset_path, 'features.dat')
feature_cache_indice_path = os.path.join(dataset_path, 'feature_indice.dat')
feature_cache_path = os.path.join(dataset_path, 'feature_cache.dat')
mmapped_features = loadingMMAPFeature(features_path, shape)
feature_dim = shape[1]

print ("="* 20 + "Loading Finished" + "=" * 20)
os.environ['GATHER_NUM_THREADS'] = str(args.gather_num_threads)

cuda_gather = gather_gpu.CUDA_Gather(features_path, feature_cache_indice_path, num_nodes, feature_dim)
cuda_gather.init_static_from_file(feature_cache_path)
# default: UVA

print ("="* 20 + "Initializing Cache Finished" + "=" * 20)
subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                  batch_size=4096, shuffle=False,
                                  num_workers=12)

# os.environ['GINEX_NUM_THREADS'] = str(args.ginex_num_threads)
csr_topo = quiver.CSRTopo(data.edge_index)

quiver_sampler = GraphSageSampler(csr_topo, sizes=[15, 10, 5], device=0, mode='UVA')


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(SAGE, self).__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

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


def superBatchSample(i, last, mode = "train"):
    global sample_time
    global ssdW_time
    
    # No changeset precomputation when i == 0
    if i != 0:
        effective_sb_size = int((train_idx.numel()%(args.sb_size*args.batch_size) + args.batch_size - 1) / args.batch_size) if last else args.sb_size

    torch.cuda.empty_cache()
    start_idx = i * args.batch_size * args.sb_size 
    end_idx = min((i+1) * args.batch_size * args.sb_size, train_idx.numel())

    train_loader = torch.utils.data.DataLoader(train_idx[start_idx:end_idx],
                                           batch_size=args.batch_size,
                                           shuffle=False)
    for step, seeds in enumerate(train_loader):
        
        sample_start = time.time()
        n_id, batch_size, adjs = quiver_sampler.sample(seeds)
        n_id_filename = os.path.join('/data01/liuyibo/trace', args.exp_name, 'sb_' + str(i) + '_ids_' + str(step) + '.pth')
        adjs_filename = os.path.join('/data01/liuyibo/trace', args.exp_name, 'sb_' + str(i) + '_adjs_' + str(step) + '.pth')
        
        if (quiver_sampler.mode != 'CPU'):
            n_id = n_id.to("cpu")
            adjs = [adj.to("cpu") for adj in adjs]
        sample_floating = time.time() - sample_start
        sample_time += sample_floating
        
        ssdW_start = time.time()
        torch.save(n_id, n_id_filename)
        torch.save(adjs, adjs_filename)
        ssdW_time += time.time() - ssdW_start

def trace_load(q, indices, sb):
    for i in indices:
        q.put((
            torch.load('/data01/liuyibo/trace/' + args.exp_name + '/' + 'sb_' + str(sb) + '_ids_' + str(i) + '.pth'),
            torch.load('/data01/liuyibo/trace/' + args.exp_name + '/' + 'sb_' + str(sb) + '_adjs_' + str(i) + '.pth'),
            ))


    

def gather(gather_q, n_id, batch_size):
    global gather_time
    
    gather_start = time.time()
    batch_inputs = cuda_gather.gather(n_id)
    batch_labels = labels[n_id[:batch_size]]
    gather_floating = time.time() - gather_start
    gather_time += gather_floating
    
    gather_q.put((batch_inputs, batch_labels))
    

def execute(pbar, total_loss, total_correct, mode='train'):
    # time recording
    global gather_time
    global transfer_time
    global train_time

    train_loader = torch.utils.data.DataLoader(train_idx,
                                           batch_size=args.batch_size,
                                           shuffle=True)
    for _, seeds in enumerate(train_loader):
        n_id, batch_size, adjs = quiver_sampler.sample(seeds)
        batch_size = adjs[-1].size[1].item()
        print(batch_size)
        if (quiver_sampler.mode != 'CPU'):
            n_id = n_id.to('cpu')
        # Gather
        gather_start = time.time()
        batch_inputs_cuda = cuda_gather.gather(n_id)
        batch_labels = labels[n_id[:batch_size]]
        gather_floating = time.time() - gather_start
        gather_time += gather_floating

        # if idx != 0:
        #     # Gather
        #     (batch_inputs, batch_labels) = gather_q.get()


        # if idx != num_iter-1:
        #     # Sample
        #     q_value = q[(idx + 1) % args.trace_load_num_threads].get()
        #     if q_value:
        #         n_id, adjs = q_value
        #         batch_size = adjs[-1].size[1].item()
        #         n_id_q.put(n_id)
        #         adjs_q.put(adjs)
            # Gather
            # gather_loader = threading.Thread(target=gather, args=(gather_q, n_id, batch_size), daemon=True)
            # gather_loader.start()

        # Transfer
        transfer_start = time.time()
        # batch_inputs_cuda = batch_inputs.to(device)
        batch_labels_cuda = batch_labels.to(device)
        adjs = [adj.to(device) for adj in adjs]
        
        torch.cuda.synchronize()
        transfer_floating = time.time() - transfer_start
        transfer_time += transfer_floating

        # Forward
        train_start = time.time()
        out = model(batch_inputs_cuda, adjs)
        loss = F.nll_loss(out, batch_labels_cuda.long())

        # Backward
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_time += time.time() - train_start
        
        # Free
        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(batch_labels_cuda.long()).sum())
        del(n_id)
        
        
        del(batch_inputs_cuda)
        del(batch_labels_cuda)
        torch.cuda.empty_cache()
        pbar.update(batch_size)

    return total_loss, total_correct

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SAGE(dataset.num_features, 256, dataset.num_classes, num_layers=3)
model = model.to(device)



y = data.y.squeeze().to(device)


def train(epoch):
    global sample_time
    global gather_time
    global transfer_time
    global train_time
    global read_time
    
    model.train()

    num_iter = int((train_idx.numel()+args.batch_size-1) / args.batch_size)

    pbar = tqdm(total=train_idx.numel(), position=0, leave=True)
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    
    total_loss, total_correct = execute(pbar, total_loss, total_correct, mode='train')
        


    pbar.close()

    loss = total_loss / num_iter
    approx_acc = total_correct / train_idx.numel()

    return loss, approx_acc


@torch.no_grad()
def test():
    model.eval()

    out = model.inference(mmapped_features)

    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    val_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, val_acc, test_acc



if __name__=='__main__':
    
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    best_val_acc = final_test_acc = 0
    for epoch in range(args.num_epochs):
        if args.verbose:
            tqdm.write('Running Epoch {}...'.format(epoch))

        start = time.time()
        loss, acc = train(epoch)
        end = time.time()
        tqdm.write(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
        tqdm.write('Epoch time: {:.4f} ms'.format((end - start) * 1000))
        
        if epoch > 5 and not args.train_only:
            train_acc, val_acc, test_acc = test()
            print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                f'Test: {test_acc:.4f}')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc
        
        tqdm.write('sample time: {:.4f} s, gather time: {:.4f} s, \
                    transfer time: {:.4f} s, train time: {:.4f} s,\
                    ssd time: {:.4f} s, \
                    read time: {:.4f} s'.format(sample_time, gather_time, \
                    transfer_time, train_time, ssdW_time, read_time))
        sample_time = 0
        gather_time = 0
        transfer_time = 0
        train_time = 0
        ssdW_time = 0
        read_time = 0
        

    