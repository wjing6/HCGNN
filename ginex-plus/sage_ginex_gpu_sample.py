import os
import sys
import json
import glob
import torch
import torch.nn.functional as F
from tqdm import tqdm
import quiver
import time
import threading
from datetime import datetime
import numpy as np
from lib.cache import NeighborCache, FeatureCache
from lib.classical_cache import FIFO
from lib.neighbor_sampler import GinexNeighborSampler
from lib.data import GinexDataset
from lib.utils import *
import argparse
import argcomplete
from model.sage_with_stale import SAGE
from queue import Queue
from log import log
import csv
sys.path.append("../")
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

losses = []
cur_time = []


# parse command line args
argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, default='ogbn-papers100M')
argparser.add_argument('--feature-cache-size', type=float, default=500000000)
argparser.add_argument('--sb-size', type=int, default=1000)
argparser.add_argument('--prop', type=float, default=0.01)
argparser.add_argument('--gpu', type=int, default=0)
# whether use GPU for sampling
argparser.add_argument('--use-gpu', dest='use_gpu',
                       default=False, action='store_true')
argparser.add_argument('--exist-binary', dest='exist_binary',
                       default=False, action='store_true')
argparser.add_argument('--num-epochs', type=int, default=10)
argparser.add_argument('--batch-size', type=int, default=1000)
argparser.add_argument('--num-workers', type=int, default=os.cpu_count())
argparser.add_argument('--stale-thre', type=int, default=5)
argparser.add_argument('--feature-dim', type=int, default=256)
argparser.add_argument('--num-hiddens', type=int, default=256)
argparser.add_argument('--exp-name', type=str, default=None)
argparser.add_argument('--sizes', type=str, default='10,10,10')
argparser.add_argument('--embedding-sizes', type=str, default='0.0005,0.001')
argparser.add_argument('--trace-load-num-threads', type=int, default=4)
argparser.add_argument('--ginex-num-threads', type=int,
                       default=os.cpu_count()*8)
argparser.add_argument('--neigh-cache-size', type=int, default=6000000000)
argparser.add_argument('--need-neigh-cache', dest='need_neigh_cache',
                       default=False, action='store_true')
argparser.add_argument('--verbose', dest='verbose',
                       default=False, action='store_true')
argparser.add_argument('--train-only', dest='train_only',
                       default=False, action='store_true')
args, remaining_args = argparser.parse_known_args()
root = "/mnt/Ginex/dataset/"
dataset_path = os.path.join(root, args.dataset + '-ginex')

if args.dataset in ['igb-medium', 'igb-large']:
    argparser.add_argument('--path', type=str, default=dataset_path,
                           help='path containing the datasets')
    argparser.add_argument('--dataset_size', type=str, default='medium',
                           choices=['tiny', 'small',
                                    'medium', 'large', 'full'],
                           help='size of the datasets')
    argparser.add_argument('--num_classes', type=int, default=19,
                           choices=[19, 2983], help='number of classes')
    argparser.add_argument('--in_memory', type=int, default=0,
                           choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
    argparser.add_argument('--synthetic', type=int, default=0,
                           choices=[0, 1], help='0:nlp-node embeddings, 1:random')

# 启用 argcomplete 自动补全
argcomplete.autocomplete(argparser)

# 解析命令行参数
args = argparser.parse_args()



split_idx_path = os.path.join(dataset_path, 'split_idx.pth')
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
os.environ['GINEX_NUM_THREADS'] = str(args.ginex_num_threads)
log.info(
    f"GPU: {str(args.gpu)}, tot available gpu: {torch.cuda.device_count()}")
sizes = [int(size) for size in args.sizes.split(',')]
dataset = GinexDataset(dataset_path, args.dataset, split_idx_path=split_idx_path)

start_time = time.time()

# if not args.use_gpu:
#     dataset.save_neighbor_cache(args.neigh_cache_size)

# edge_index = dataset.edge_index
# log.info(f"edge_index: {edge_index.shape}")
embedding_rate = [float(size) for size in args.embedding_sizes.split(',')]
embedding_sizes = [int(rate * dataset.num_nodes) for rate in embedding_rate]
trace_path = './trace'

if args.exp_name is None:
    now = datetime.now()
    args.exp_name = now.strftime('%Y_%m_%d_%H_%M_%S')
os.makedirs(os.path.join(trace_path, args.exp_name), exist_ok=True)

device = torch.device('cuda:%d' % args.gpu)
torch.cuda.set_device(device)
# TODO: use OmegaConf to reduce parameter
model = SAGE(
    in_channels=dataset.feature_dim,
    hidden_channels=args.num_hiddens,
    out_channels=dataset.num_classes,
    num_layers=len(sizes),
    num_nodes=dataset.num_nodes,
    device=device,
    cache_device=device,
    stale_thre=args.stale_thre,
    embedding_rate=embedding_rate
)
model = model.to(device)

mmapped_features = dataset.get_mmapped_features()
log.info("loading feature finish")
num_nodes = dataset.num_nodes
feature_dim = dataset.feature_dim
feature_path = dataset.features_path
labels = dataset.get_labels()
log.info("loading labels finish")
log.info(f"{args.use_gpu}")
if args.use_gpu:
    indptr, indices = dataset.get_adj_mat_direct()
else:
    indptr, indices = dataset.get_adj_mat()
log.info("loading indptr and indice finish, all prepare finish")


def get_csr(edge_index):
    transfer_start = time.time()
    csr_topo = quiver.CSRTopo(indices=indices, indptr=indptr)
    log.info(f"csr topo cost: {time.time() - transfer_start}s")
    return csr_topo

def get_sampler(indptr, indices, embedding_cache):
    """构建Sampler，目前只有在GPU采样器情况下调用此函数"""
    if args.use_gpu:
        transfer_start = time.time()
        csr_topo = quiver.CSRTopo(indices=indices, indptr=indptr)
        log.info(f"csr topo cost: {time.time() - transfer_start}s")
        neigh_sampler = quiver.pyg.GraphSageSampler(
            csr_topo, dataset.num_nodes, exp_name=args.exp_name, trace_path=trace_path,
                sizes=sizes, embedding_cache=embedding_cache, device=args.gpu, mode='UVA')
        log.info(f"indptr size: {csr_topo.indptr.size()}")
        log.info(f"indptr: {csr_topo.indptr[-5:]}")
        log.info(f"indice: {csr_topo.indices[-5:]}")
    else:
        """NEVER REACHES HERE!"""
        assert False, "get_sampler should only be called in inspect_gpu"
        '''
        Now use prop for test, so we can set 'cache_size' as 0
        use '1' for all neighbor placed in CPU
        '''
        indptr = dataset.get_rowptr_mt()
        score = dataset.get_score()
        num_nodes = dataset.num_nodes

        neighbor_cache = NeighborCache(
            size=0,
            score=score,
            indptr=indptr,
            indices=dataset.indices_path,
            num_nodes=num_nodes,
            prop=1
        )

        neigh_sampler = GinexNeighborSampler(indptr, dataset.indices_path, node_idx=dataset.shuffled_train_idx,
                                       sizes=sizes, num_nodes = num_nodes,
                                       cache_data = neighbor_cache.cache, address_table = neighbor_cache.address_table,
                                       batch_size=args.batch_size,
                                       shuffle=False)
        #FIXME: 实际上好像也不需要返回 CPU 采样器..
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return neigh_sampler


def inspect_cpu(i, last, mode='train'):
    # with open('/proc/sys/vm/drop_caches', 'w') as stream:
    #     stream.write('1\n')

    # TODO: 统一调用方式
    if mode == 'train':
        node_idx = dataset.shuffled_train_idx
    elif mode == 'valid':
        node_idx = dataset.val_idx
    elif mode == 'test':
        node_idx = dataset.test_idx
    if i != 0:
        effective_sb_size = int((node_idx.numel() % (
            args.sb_size*args.batch_size) + args.batch_size-1) / args.batch_size) if last else args.sb_size
        cache = FeatureCache(args.feature_cache_size, effective_sb_size, num_nodes,
                             mmapped_features, feature_dim, args.exp_name, trace_path, i - 1, args.verbose, False)
        # Pass 1 and 2 are executed before starting sb sample.
        # We overlap only the pass 3 of changeset precomputation,
        # which is the most time consuming part, with sb sample.
        iterptr, iters, initial_cache_indices = cache.pass_1_and_2()

        # Only changset precomputation at the last superbatch in epoch
        if last:
            cache.pass_3(iterptr, iters, initial_cache_indices)
            torch.cuda.empty_cache()
            return cache, initial_cache_indices.cpu(), 0
        else:
            torch.cuda.empty_cache()

    # 加载neighbor cache和neighbor cachetable
    neighbor_cache_path = str(dataset_path) + '/nc_size_' + str(args.neigh_cache_size) + '.dat'
    neighbor_cache_conf_path = str(
        dataset_path) + '/nc_size_' +  str(args.neigh_cache_size) + '_conf.json'
    neighbor_cache_numel = json.load(
        open(neighbor_cache_conf_path, 'r'))['shape'][0]
    neighbor_cachetable_path = str(
        dataset_path) + '/nctbl_size_' + str(args.neigh_cache_size) + '.dat'
    neighbor_cachetable_conf_path = str(
        dataset_path) + '/nctbl_size_' + str(args.neigh_cache_size) + '_conf.json'
    neighbor_cachetable_numel = json.load(
        open(neighbor_cachetable_conf_path, 'r'))['shape'][0]
    

    neighbor_cache = load_int64(neighbor_cache_path, neighbor_cache_numel)
    neighbor_cachetable = load_int64(
        neighbor_cachetable_path, neighbor_cachetable_numel)
    

    start_idx = i * args.batch_size * args.sb_size
    end_idx = min((i+1) * args.batch_size * args.sb_size, node_idx.numel())
    loader = GinexNeighborSampler(
        indptr=indptr,
        indices=dataset.indices_path,
        exp_name=args.exp_name,
        sb=i,
        staleness_thre=args.stale_thre,
        sizes=sizes,
        node_idx=node_idx[start_idx:end_idx],
        embedding_size=embedding_rate,
        cache_data=neighbor_cache,
        address_table=neighbor_cachetable,
        num_nodes=num_nodes,
        cache_dim=args.num_hiddens,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16 # originally 0 ?
    )

    start = time.time()
    for step, _ in enumerate(loader):
        if i != 0 and step == 0:
            cache.pass_3(iterptr, iters, initial_cache_indices)
    sample_time = time.time() - start
    tensor_free(neighbor_cache)
    tensor_free(neighbor_cachetable)

    if i != 0:
        return cache, initial_cache_indices.cpu(), sample_time
    else:
        return None, None, sample_time


def inspect_gpu(i, last, gpu_sampler, mode='train'):
    if mode == 'train':
        shuffled_train_idx = dataset.shuffled_train_idx
    elif mode == 'valid':
        shuffled_train_idx = dataset.val_idx
    elif mode == 'test':
        shuffled_train_idx = dataset.test_idx


    sampler_start = time.time()

    if i != 0:
        effective_sb_size = int((shuffled_train_idx.numel() % (
            args.sb_size*args.batch_size) + args.batch_size-1) / args.batch_size) if last else args.sb_size
        cache = FeatureCache(args.feature_cache_size, effective_sb_size, num_nodes,
                             mmapped_features, feature_dim, args.exp_name, trace_path, i - 1, args.verbose, False)
        iterptr, iters, initial_cache_indices = cache.pass_1_and_2()
        if last:
            cache.pass_3(iterptr, iters, initial_cache_indices)
            torch.cuda.empty_cache()
            log.info(f"superbatch {i}, hit num: {cache.hit_num}, total place: {cache.total_place}\
                     , hit ratio: {cache.hit_num / cache.total_place:.4f}")
            return cache, initial_cache_indices.cpu(), time.time() - sampler_start, 0
        else:
            torch.cuda.empty_cache()
    start_idx = i * args.batch_size * args.sb_size
    end_idx = min((i+1) * args.batch_size * args.sb_size, shuffled_train_idx.numel())
    train_loader = torch.utils.data.DataLoader(shuffled_train_idx[start_idx:end_idx],
                                                   batch_size=args.batch_size,
                                                   shuffle=False)
    start = time.time()
    for _, mini_batch_seeds in enumerate(train_loader):
        gpu_sampler.sample(mini_batch_seeds)
    sample_time = time.time() - start
    if gpu_sampler.embedding_cache is not None: # training stage
        for layer, fifo_cache in gpu_sampler.embedding_cache.items():
            if (fifo_cache.total_place_num > 0):
                log.info(f"Layer: {layer}, hit number: {fifo_cache.hit_num}, \
                    total place: {fifo_cache.total_place_num}, hit ratio: {(fifo_cache.hit_num / fifo_cache.total_place_num):.4f}")
    if i != 0:
        cache.pass_3(iterptr, iters, initial_cache_indices)
        torch.cuda.synchronize()
        log.info(f"superbatch {i}, hit num: {cache.hit_num}, total place: {cache.total_place}\
                     , hit ratio: {cache.hit_num / cache.total_place:.4f}")
        return cache, initial_cache_indices.cpu(), time.time() - sampler_start, sample_time
    else:
        torch.cuda.synchronize()
        return None, None, time.time() - sampler_start, sample_time


def switch(cache, initial_cache_indices):
    cache.fill_cache(initial_cache_indices)
    del (initial_cache_indices)
    return cache


def trace_load(q, indices, sb):
    for i in indices:
        q.put((
            torch.load(trace_path + '/' + args.exp_name + '/' + 'sb_' +
                       str(sb) + '_ids_' + str(i) + '.pth'),
            torch.load(trace_path + '/' + args.exp_name + '/' + 'sb_' +
                       str(sb) + '_adjs_' + str(i) + '.pth'),
            torch.load(trace_path + '/' + args.exp_name + '/' + 'sb_' +
                       str(sb) + '_update_' + str(i) + '.pth'),
        ))


def gather(gather_q, n_id, cache, batch_size):
    batch_inputs, _ = gather_ginex(feature_path, n_id, feature_dim, cache)
    batch_labels = labels[n_id[:batch_size]]
    gather_q.put((batch_inputs, batch_labels, n_id))


def delete_trace(i):
    n_id_filelist = glob.glob(
        trace_path + '/' + args.exp_name + '/sb_' + str(i - 1) + '_ids_*')
    adjs_filelist = glob.glob(
        trace_path + '/' + args.exp_name + '/sb_' + str(i - 1) + '_adjs_*')
    cache_filelist = glob.glob(
        trace_path + '/' + args.exp_name + '/sb_' + str(i - 1) + '_update_*')

    for n_id_file in n_id_filelist:
        try:
            os.remove(n_id_file)
        except:
            tqdm.write('Error while deleting file : ', n_id_file)

    for adjs_file in adjs_filelist:
        try:
            os.remove(adjs_file)
        except:
            tqdm.write('Error while deleting file : ', adjs_file)

    for cache_file in cache_filelist:
        try:
            os.remove(cache_file)
        except:
            tqdm.write('Error while deleting file : ', cache_file)

# 因为 gather 是从 feature cache 中获取数据, 所以不需要对 gather 进行修改
# 但在 forward 过程中, 需要将 embedding 中的数据拼接输入到下一层


def execute(i, cache, pbar, total_loss, total_correct, last, mode='train'):
    if last:
        if mode == 'train':
            num_iter = int((dataset.shuffled_train_idx.numel() % (
                args.sb_size*args.batch_size) + args.batch_size-1) / args.batch_size)
            log.info(f"start {num_iter} execute")
        elif mode == 'valid':
            num_iter = int((dataset.val_idx.numel() % (
                args.sb_size*args.batch_size) + args.batch_size-1) / args.batch_size)
        elif mode == 'test':
            num_iter = int((dataset.test_idx.numel() % (
                args.sb_size*args.batch_size) + args.batch_size-1) / args.batch_size)
    else:
        num_iter = args.sb_size

    # Multi-threaded load of sets of (ids, adj, update)
    q = list()
    loader = list()

    # When execute, need refresh the cache !!
    model.reset_embeddings()

    for t in range(args.trace_load_num_threads):
        q.append(Queue(maxsize=2))
        loader.append(threading.Thread(target=trace_load, args=(q[t], list(
            range(t, num_iter, args.trace_load_num_threads)), i-1), daemon=True))
        loader[t].start()

    n_id_q = Queue(maxsize=2)
    adjs_q = Queue(maxsize=2)
    in_indices_q = Queue(maxsize=2)
    in_positions_q = Queue(maxsize=2)
    out_indices_q = Queue(maxsize=2)
    gather_q = Queue(maxsize=1)

    training_time = 0

    for idx in range(num_iter):
        batch_size = args.batch_size
        if idx == 0:
            # Sample
            q_value = q[idx % args.trace_load_num_threads].get()
            if q_value:
                n_id, adjs, (in_indices, in_positions, out_indices) = q_value
                log.info(
                    f"loading n_id: {n_id.shape} finish, {adjs}, {in_indices.shape}, {in_positions.shape}, {out_indices.shape}")
                batch_size = adjs[-1].size[1]
                n_id_q.put(n_id)
                adjs_q.put(adjs)
                in_indices_q.put(in_indices)

                in_positions_q.put(in_positions)
                out_indices_q.put(out_indices)

            # Gather
            batch_inputs, _ = gather_ginex(
                feature_path, n_id, feature_dim, cache)
            log.info(
                f"loading batch input finish, input shape: {batch_inputs.shape}")
            batch_labels = labels[n_id[:batch_size]]

            # Cache
            cache.update(batch_inputs, in_indices, in_positions, out_indices)

        if idx != 0:
            # Gather
            (batch_inputs, batch_labels, n_id) = gather_q.get()

            # Cache
            in_indices = in_indices_q.get()
            in_positions = in_positions_q.get()
            out_indices = out_indices_q.get()
            cache.update(batch_inputs, in_indices, in_positions, out_indices)

        if idx != num_iter-1:
            # Sample
            q_value = q[(idx + 1) % args.trace_load_num_threads].get()
            if q_value:
                n_id, adjs, (in_indices, in_positions, out_indices) = q_value
                batch_size = adjs[-1].size[1]
                n_id_q.put(n_id)
                adjs_q.put(adjs)
                in_indices_q.put(in_indices)
                in_positions_q.put(in_positions)
                out_indices_q.put(out_indices)

            # Gather
            gather_loader = threading.Thread(target=gather, args=(
                gather_q, n_id, cache, batch_size), daemon=True)
            gather_loader.start()

        start = time.time()
        # Transfer
        batch_inputs_cuda = batch_inputs.to(device)
        batch_labels_cuda = batch_labels.to(device)
        adjs_host = adjs_q.get()
        adjs = [adj.to(device) for adj in adjs_host]

        # Forward
        n_id = n_id_q.get()
        n_id_cuda = n_id.to(device)
        out = model(batch_inputs_cuda, adjs, n_id_cuda)
        loss = F.nll_loss(out, batch_labels_cuda.long())
        losses.append(loss.item())
        cur_time.append(time.time() - start_time)
        # Backward
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        training_time += time.time() - start

        # Free
        total_loss += float(loss)
        correct_in_batch = int(out.argmax(
            dim=-1).eq(batch_labels_cuda.long()).sum())
        total_correct += correct_in_batch
        res = [idx, loss, float(correct_in_batch / batch_labels_cuda.shape[0])]

        del (n_id)
        if idx == 0:
            in_indices = in_indices_q.get()
            in_positions = in_positions_q.get()
            out_indices = out_indices_q.get()
        del (in_indices)
        del (in_positions)
        del (out_indices)
        del (adjs_host)
        tensor_free(batch_inputs)
        pbar.update(batch_size)
    log.info(f"epoch: {epoch:02d}, evit time: {model.evit_time}, index select time: {model.index_select_time}, cache transfer time: {model.cache_transfer_time}")
    # not cache pre-epoch embeddings

    return total_loss, total_correct, training_time


def train_sample_cpu(epoch):
    model.train()
    inspect_time = 0
    neighbor_indice_time = 0
    training_time = 0
    gather_and_train_time = 0

    dataset.make_new_shuffled_train_idx()
    num_iter = int((dataset.shuffled_train_idx.numel() +
                   args.batch_size-1) / args.batch_size)

    pbar = tqdm(total=dataset.shuffled_train_idx.numel(), position=0, leave=True)
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    num_sb = int((dataset.shuffled_train_idx.numel()+args.batch_size *
                 args.sb_size-1)/(args.batch_size*args.sb_size))

    for i in range(num_sb + 1):
        if args.verbose:
            tqdm.write(
                'Running {}th superbatch of total {} superbatches'.format(i, num_sb))

        # Superbatch sample
        if args.verbose:
            tqdm.write('Step 1: Superbatch Sample')
        start = time.time()
        cache, initial_cache_indices, sampler_batch_time = inspect_cpu(
            i, 
            last=(i == num_sb), 
            mode='train'
        )
        torch.cuda.synchronize()
        inspect_time += time.time() - start
        neighbor_indice_time += sampler_batch_time

        if args.verbose:
            tqdm.write('Step 1: Done')

        if i == 0:
            continue

        # Switch
        if args.verbose:
            tqdm.write('Step 2: Switch')
        cache = switch(cache, initial_cache_indices)
        torch.cuda.synchronize()
        if args.verbose:
            tqdm.write('Step 2: Done')

        # Main loop
        if args.verbose:
            tqdm.write('Step 3: Main Loop')
        start = time.time()
        total_loss, total_correct, cur_train_time = execute(
            i, 
            cache, 
            pbar, 
            total_loss, 
            total_correct, 
            last=(i == num_sb), 
            mode='train'
        )
        training_time += cur_train_time
        gather_and_train_time += time.time() - start
        if args.verbose:
            tqdm.write('Step 3: Done')

        # Delete obsolete runtime files
        delete_trace(i)
        
    pbar.close()

    loss = total_loss / num_iter
    approx_acc = total_correct / dataset.shuffled_train_idx.numel()

    
    return loss, approx_acc, gather_and_train_time, training_time, inspect_time, neighbor_indice_time


def train_sample_gpu(epoch, gpu_sampler):
    model.train()
    inspect_time = 0
    sampling_time = 0
    gather_and_train_time = 0
    training_time = 0

    dataset.make_new_shuffled_train_idx()
    shuffled_train_idx = dataset.shuffled_train_idx
    num_iter = int((shuffled_train_idx.numel() +
                   args.batch_size-1) / args.batch_size)

    pbar = tqdm(total=shuffled_train_idx.numel(), position=0, leave=True)
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    num_sb = int((shuffled_train_idx.numel()+args.batch_size *
                 args.sb_size-1)/(args.batch_size*args.sb_size))

    for i in range(num_sb + 1):
        if args.verbose:
            tqdm.write(
                'Running {}th superbatch of total {} superbatches'.format(i, num_sb))

        # Superbatch sample
        if args.verbose:
            tqdm.write('Step 1: Superbatch Sample')
        cache, initial_cache_indices, sampler_batch_time, exact_sample_time = inspect_gpu(
            i, last=(i == num_sb), gpu_sampler=gpu_sampler, mode='train')
        if args.verbose:
            tqdm.write('Step 1: Done')
        inspect_time += sampler_batch_time
        sampling_time += exact_sample_time

        if i == 0:
            gpu_sampler.inc_sb()
            gpu_sampler.fresh_embedding()
            continue

        # Switch
        if args.verbose:
            tqdm.write('Step 2: Switch')
        cache = switch(cache, initial_cache_indices)
        torch.cuda.synchronize()
        if args.verbose:
            tqdm.write('Step 2: Done')

        # Main loop
        if args.verbose:
            tqdm.write('Step 3: Main Loop')
        start = time.time()
        total_loss, total_correct, train_time = execute(
            i, cache, pbar, total_loss, total_correct, last=(i == num_sb), mode='train')
        gather_and_train_time += time.time() - start
        training_time += train_time
        if args.verbose:
            tqdm.write('Step 3: Done')

        # Delete obsolete runtime files
        delete_trace(i)
        # TODO: refine it!
        gpu_sampler.inc_sb()
        gpu_sampler.fresh_embedding()

    pbar.close()

    loss = total_loss / num_iter
    approx_acc = total_correct / shuffled_train_idx.numel()
    return loss, approx_acc, gather_and_train_time, training_time, inspect_time, sampling_time


@torch.no_grad()
def inference_gpu_sample(gpu_sampler, mode='test'):
    model.eval()

    if mode == 'test':
        pbar = tqdm(total=dataset.test_idx.numel(), position=0, leave=True)
        num_sb = int((dataset.test_idx.numel()+args.batch_size *
                     args.sb_size-1)/(args.batch_size*args.sb_size))
        num_iter = int((dataset.test_idx.numel() +
                       args.batch_size-1) / args.batch_size)
    elif mode == 'valid':
        pbar = tqdm(total=dataset.val_idx.numel(), position=0, leave=True)
        num_sb = int((dataset.val_idx.numel()+args.batch_size *
                     args.sb_size-1)/(args.batch_size*args.sb_size))
        num_iter = int(
            (dataset.val_idx.numel()+args.batch_size-1) / args.batch_size)

    pbar.set_description('Evaluating')

    model.change_stage(1)

    total_loss = total_correct = 0
    for i in range(num_sb + 1):
        if args.verbose:

            tqdm.write(
                'Running {}th superbatch of total {} superbatches'.format(i, num_sb))

        # Superbatch sample
        if args.verbose:
            tqdm.write('Step 1: Superbatch Sample')
        cache, initial_cache_indices, _, _ = inspect_gpu(
            i, last=(i == num_sb), gpu_sampler=gpu_sampler, mode=mode)
        torch.cuda.synchronize()
        if args.verbose:
            tqdm.write('Step 1: Done')

        if i == 0:
            gpu_sampler.inc_sb()
            continue

        # Switch
        if args.verbose:
            tqdm.write('Step 2: Switch')
        cache = switch(cache, initial_cache_indices)
        torch.cuda.synchronize()
        if args.verbose:
            tqdm.write('Step 2: Done')

        # Main loop
        if args.verbose:
            tqdm.write('Step 3: Main Loop')
        total_loss, total_correct, _ = execute(
            i, cache, pbar, total_loss, total_correct, last=(i == num_sb), mode=mode)
        if args.verbose:
            tqdm.write('Step 3: Done')

        # Delete obsolete runtime files
        delete_trace(i)
        gpu_sampler.inc_sb()

    pbar.close()

    loss = total_loss / num_iter
    if mode == 'test':
        approx_acc = total_correct / dataset.test_idx.numel()
    elif mode == 'valid':
        approx_acc = total_correct / dataset.val_idx.numel()
    
    model.change_stage(0) # 恢复训练状态
    gpu_sampler.reset_sampler()

    return loss, approx_acc

@torch.no_grad()
def inference(mode='test'):
    model.eval()

    if mode == 'test':
        pbar = tqdm(total=dataset.test_idx.numel(), position=0, leave=True)
        num_sb = int((dataset.test_idx.numel()+args.batch_size *
                     args.sb_size-1)/(args.batch_size*args.sb_size))
        num_iter = int((dataset.test_idx.numel() +
                       args.batch_size-1) / args.batch_size)
    elif mode == 'valid':
        pbar = tqdm(total=dataset.val_idx.numel(), position=0, leave=True)
        num_sb = int((dataset.val_idx.numel()+args.batch_size *
                     args.sb_size-1)/(args.batch_size*args.sb_size))
        num_iter = int(
            (dataset.val_idx.numel()+args.batch_size-1) / args.batch_size)

    pbar.set_description('Evaluating')

    total_loss = total_correct = 0

    for i in range(num_sb + 1):
        if args.verbose:

            tqdm.write(
                'Running {}th superbatch of total {} superbatches'.format(i, num_sb))

        # Superbatch sample
        if args.verbose:
            tqdm.write('Step 1: Superbatch Sample')
        cache, initial_cache_indices, _ = inspect_cpu(
            i, last=(i == num_sb), mode=mode)
        torch.cuda.synchronize()
        if args.verbose:
            tqdm.write('Step 1: Done')

        if i == 0:
            continue

        # Switch
        if args.verbose:
            tqdm.write('Step 2: Switch')
        cache = switch(cache, initial_cache_indices)
        torch.cuda.synchronize()
        if args.verbose:
            tqdm.write('Step 2: Done')

        # Main loop
        if args.verbose:
            tqdm.write('Step 3: Main Loop')
        total_loss, total_correct, _ = execute(
            i, cache, pbar, total_loss, total_correct, last=(i == num_sb), mode=mode)
        if args.verbose:
            tqdm.write('Step 3: Done')

        # Delete obsolete runtime files
        delete_trace(i)

    pbar.close()

    loss = total_loss / num_iter
    if mode == 'test':
        approx_acc = total_correct / dataset.test_idx.numel()
    elif mode == 'valid':
        approx_acc = total_correct / dataset.val_idx.numel()

    return loss, approx_acc


if __name__ == '__main__':
    log.info("enter training process")
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    best_val_acc = final_test_acc = 0

    if len(embedding_rate) != len(sizes) - 1:
        raise ValueError('Embedding layer excludes the training node and \
                the bottom feature, expected sizes of length {} but found {}'.format(
            len(sizes) - 1, len(embedding_rate)))

    embedding_cache = {}
    for i in range(1, len(sizes)):
        # init the embedding cache
        # what is the layer_1 ? top down !
        tmp_tag = 'layer_' + str(i)
        embedding_cache[tmp_tag] = FIFO(num_nodes, tmp_tag, args.num_hiddens,
                                        args.stale_thre, fifo_ratio=embedding_rate[i - 1], device=device)
    log.info("Initialize embedding cache finish")

    total_batch = int((dataset.train_idx.numel() + args.batch_size - 1) / args.batch_size)
    avg_time = 0 #记录平均训练时间

    if args.use_gpu:
        log.info("initialize gpu sampler..")
        gpu_sampler = get_sampler(indptr, indices, embedding_cache)
        if args.dataset in ["ogbn-papers100M", "ogbn-products", "igb-medium"]:
            inference_indptr, inference_indices = dataset.get_adj_mat()
            inference_gpu_sampler = get_sampler(inference_indptr, inference_indices, None) # inference 不使用嵌入缓存
        for epoch in range(args.num_epochs):
            if args.verbose:
                tqdm.write('\n==============================')
                tqdm.write('Running Epoch {}...'.format(epoch))

            start = time.time()
            loss, acc, gather_and_train_time, training_time, inspect_time, sample_time = train_sample_gpu(epoch, gpu_sampler)
            end = time.time()
            log.info(
                f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
            log.info('Epoch time: {:.4f} s, Train time: {:.4f} s, Gather time: {:.4f} s, Inspect time: {:.4f} s\
                     '.format((end - start), training_time, (gather_and_train_time - training_time), inspect_time))
            log.info('Sample time: {:.4f} s, Save time:{:.4f}'.format(sample_time, gpu_sampler.save_time))
            log.info('average reduce: {:.4f}, average after: {:.4f}'.format(gpu_sampler.reduce / total_batch, gpu_sampler.after / total_batch))
            for i in range(1, len(sizes)):
                tmp_tag = 'layer_' + str(i)
                log.info(f"cache fresh time: {embedding_cache[tmp_tag].cache_fresh_time} s")

            avg_time = avg_time+(end-start)*1000

            if epoch > 6 and not args.train_only and args.dataset in ["ogbn-papers100M", "ogbn-products", "igb-medium"]:

                val_loss, val_acc = inference_gpu_sample(inference_gpu_sampler, mode='valid')
                test_loss, test_acc = inference_gpu_sample(inference_gpu_sampler, mode='test')
                log.info('Valid loss: {0:.4f}, Valid acc: {1:.4f}, Test loss: {2:.4f}, Test acc: {3:.4f},'.format(
                    val_loss, val_acc, test_loss, test_acc))

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    final_test_acc = test_acc
            gpu_sampler.reset_sampler()
        avg_time = avg_time / args.num_epochs

        with open(args.dataset + '_loss.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(losses)
        with open(args.dataset + '_time.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(cur_time)

        if not args.train_only:
            log.info(f'Final Test acc: {final_test_acc:.4f}')
            log.info(f'Average training time is {avg_time:.4f}')

            with open('./result.txt','a') as file:
                file.write(f'stale threshold: {args.stale_thre}, embedding size: {args.embedding_sizes} .\n')
                file.write(f'Final Test acc: {final_test_acc:.4f}.\n')
                file.write(f'Average training time is {avg_time:.4f}')
    else:
        # use CPU
        for epoch in range(args.num_epochs):
            if args.verbose:
                tqdm.write('\n==============================')
                tqdm.write('Running Epoch {}...'.format(epoch))

            start = time.time()
            loss, acc, gather_and_train_time, training_time, inspect_time, sample_time = train_sample_cpu(epoch)
            end = time.time()
            log.info(
                f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
            log.info('Epoch time: {:.4f} s, Train time: {:.4f} s, Gather time: {:.4f} s, Inspect time: {:.4f} s\
                     '.format((end - start), training_time, (gather_and_train_time - training_time), inspect_time))
            for i in range(1, len(sizes)):
                tmp_tag = 'layer_' + str(i)
                log.info(f"cache fresh time: {embedding_cache[tmp_tag].cache_fresh_time} s")

            avg_time = avg_time + (end - start) * 1000

            if epoch > 6 and not args.train_only and args.dataset in ["ogbn-papers100M", "ogbn-products", "igb-medium"]:

                val_loss, val_acc = inference(mode='valid')
                test_loss, test_acc = inference(mode='test')
                log.info('Valid loss: {0:.4f}, Valid acc: {1:.4f}, Test loss: {2:.4f}, Test acc: {3:.4f},'.format(
                    val_loss, val_acc, test_loss, test_acc))

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    final_test_acc = test_acc
        avg_time = avg_time / args.num_epochs

        with open(args.dataset + '_loss.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(losses)
        with open(args.dataset + '_time.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(cur_time)

        if not args.train_only:
            log.info(f'Final Test acc: {final_test_acc:.4f}')
            log.info(f'Average training time is {avg_time:.4f}')

            timestamp = time.time()
            readable_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            # 追加日志
            with open('./result.txt','a') as file:
                file.write(f'==== training task ended at {readable_time} ====\n')
                file.write(f'origin command: {sys.argv}\n')
                file.write(f'stale threshold: {args.stale_thre}, embedding size: {args.embedding_sizes} .\n')
                file.write(f'Final Test acc: {final_test_acc:.4f}.\n')
                file.write(f'Average training time is {avg_time:.4f}\n')

