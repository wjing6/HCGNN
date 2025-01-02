from cmath import exp
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
from ogb.nodeproppred import PygNodePropPredDataset
from lib.utils import *
import argparse
from model.model import SAGE
from queue import Queue
from random import sample
from log import log
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


batch_for_dataset = {"ogbn-papers100M": 4096,
                     "ogbn-products": 1024,
                     "friendster": 4096,
                     "igb-medium": 4096,
                     "twitter": 4096,
                     "bytedata_caijing": 4096,
                     "bytedata_part": 4096,
                     "douyin_fengkong_guoqing_0709": 4096,
                     "douyin_fengkong_sucheng_0813": 4096,
                     "caijing_xiaowei_wangzhenchao": 4096,
                     }

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, default='ogbn-papers100M')
argparser.add_argument('--feature-cache-size', type=float, default=500000000)
argparser.add_argument('--sb-size', type=int, default=1000)
argparser.add_argument('--prop', type=float, default=0.01)
argparser.add_argument('--gpu', type=int, default=0)
# whether use GPU for sampling
argparser.add_argument('--use-gpu', type=bool, default=True)
argparser.add_argument('--exist-binary', dest='exist_binary',
                       default=False, action='store_true')
argparser.add_argument('--num-epochs', type=int, default=10)
argparser.add_argument('--batch-size', type=int, default=1000)
argparser.add_argument('--num-workers', type=int, default=os.cpu_count()*2)
argparser.add_argument('--stale-thre', type=int, default=5) # 实际未使用
argparser.add_argument('--feature-dim', type=int, default=256)
argparser.add_argument('--num-hiddens', type=int, default=256)
argparser.add_argument('--exp-name', type=str, default=None)
argparser.add_argument('--sizes', type=str, default='10,10,10')
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
args = argparser.parse_args()

root = "/data01/liuyibo/"
dataset_path = os.path.join(root, args.dataset)
split_idx_path = os.path.join(dataset_path, 'split_idx.pth')
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
os.environ['GINEX_NUM_THREADS'] = str(args.ginex_num_threads)
log.info(
    f"GPU: {str(args.gpu)}, tot available gpu: {torch.cuda.device_count()}")
sizes = [int(size) for size in args.sizes.split(',')]
dataset = GinexDataset(root, args.dataset, split_idx_path=split_idx_path)
edge_index = dataset.edge_index
log.info(f"edge_index: {edge_index.shape}")\

if args.exp_name is None:
    now = datetime.now()
    args.exp_name = now.strftime('%Y_%m_%d_%H_%M_%S')
os.makedirs(os.path.join('./trace', args.exp_name), exist_ok=True)

device = torch.device('cuda:%d' % args.gpu)
torch.cuda.set_device(device)
# TODO: use OmegaConf to reduce parameter
model = SAGE(dataset.feature_dim, args.num_hiddens,
             dataset.num_classes, num_layers=len(sizes))
model = model.to(device)

mmapped_features = dataset.get_mmapped_features()
log.info("loading feature finish")
num_nodes = dataset.num_nodes
feature_dim = dataset.feature_dim
feature_path = dataset.features_path

labels = dataset.get_labels()
log.info("loading labels finish")
indptr, indices = dataset.get_adj_mat()
log.info("loading indptr and indice finish, all prepare finish")


def get_sampler(edge_index):
    if args.use_gpu:
        transfer_start = time.time()
        log.info(f"{edge_index.shape}")
        csr_topo = quiver.CSRTopo(edge_index)
        log.info(f"csr topo cost: {time.time() - transfer_start}s")
        neigh_sampler = quiver.pyg.GraphSageSampler(
            csr_topo, dataset.num_nodes, exp_name=args.exp_name, sizes=sizes, device=args.gpu, mode='UVA')
        log.info(f"indptr size: {csr_topo.indptr.size()}")
        log.info(f"indptr: {csr_topo.indptr[-5:]}")
        log.info(f"indice: {csr_topo.indices[-5:]}")
    else:
        '''
        Now use prop for test, so we can set 'cache_size' as 0
        use '1' for all neighbor placed in CPU
        '''
        indptr = dataset.get_rowptr_mt()
        score = dataset.get_score()
        num_nodes = dataset.num_nodes
        neighbor_cache = NeighborCache(
            0, score, indptr, dataset.indices_path, num_nodes, 1)
        neigh_sampler = GinexNeighborSampler(indptr, dataset.indices_path, node_idx=dataset.shuffled_train_idx,
                                             sizes=sizes, num_nodes=num_nodes,
                                             cache_data=neighbor_cache.cache, address_table=neighbor_cache.address_table,
                                             batch_size=batch_for_dataset[args.dataset],
                                             shuffle=False)
        # FIXME: 实际上好像也不需要返回 CPU 采样器..
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return neigh_sampler


def inspect_cpu(i, last, mode='train'):
    # with open('/proc/sys/vm/drop_caches', 'w') as stream:
    #     stream.write('1\n')

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
                             mmapped_features, feature_dim, args.exp_name, i - 1, args.verbose, False)
        iterptr, iters, initial_cache_indices = cache.pass_1_and_2()

        # Only changset precomputation at the last superbatch in epoch
        if last:
            cache.pass_3(iterptr, iters, initial_cache_indices)
            torch.cuda.empty_cache()
            return cache, initial_cache_indices.cpu()
        else:
            torch.cuda.empty_cache()

    neighbor_cache_path = str(dataset_path) + '/nc_all.dat'
    neighbor_cache_conf_path = str(
        dataset_path) + '/nc_all' + '_conf.json'
    neighbor_cache_numel = json.load(
        open(neighbor_cache_conf_path, 'r'))['shape'][0]
    neighbor_cachetable_path = str(
        dataset_path) + '/nctbl_all' + '.dat'
    neighbor_cachetable_conf_path = str(
        dataset_path) + '/nctbl_all' + '_conf.json'
    neighbor_cachetable_numel = json.load(
        open(neighbor_cachetable_conf_path, 'r'))['shape'][0]
    neighbor_cache = load_int64(neighbor_cache_path, neighbor_cache_numel)
    neighbor_cachetable = load_int64(
        neighbor_cachetable_path, neighbor_cachetable_numel)
    start_idx = i * args.batch_size * args.sb_size
    end_idx = min((i+1) * args.batch_size * args.sb_size, node_idx.numel())
    loader = GinexNeighborSampler(indptr, dataset.indices_path, args.exp_name, i, args.stale_thre, node_idx=node_idx[start_idx:end_idx],
                                  sizes=sizes,
                                  cache_data=neighbor_cache, address_table=neighbor_cachetable,
                                  num_nodes=num_nodes,
                                  cache_dim=args.num_hiddens,
                                  batch_size=args.batch_size,
                                  shuffle=False, num_workers=0)

    for step, _ in enumerate(loader):
        if i != 0 and step == 0:
            cache.pass_3(iterptr, iters, initial_cache_indices)

    tensor_free(neighbor_cache)
    tensor_free(neighbor_cachetable)

    if i != 0:
        return cache, initial_cache_indices.cpu()
    else:
        return None, None


def inspect_gpu(i, last, gpu_sampler, mode='train'):
    log.info("gpu sampler start")
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
                             mmapped_features, feature_dim, args.exp_name, i - 1, args.verbose, False)
        iterptr, iters, initial_cache_indices = cache.pass_1_and_2()
        if last:
            cache.pass_3(iterptr, iters, initial_cache_indices)
            torch.cuda.empty_cache()
            return cache, initial_cache_indices.cpu(), 0
        else:
            torch.cuda.empty_cache()
    sampler_start = time.time()
    start_idx = i * args.batch_size * args.sb_size
    end_idx = min((i+1) * args.batch_size * args.sb_size, node_idx.numel())
    log.info(f"{node_idx[start_idx:end_idx].shape}")
    train_loader = torch.utils.data.DataLoader(node_idx[start_idx:end_idx],
                                               batch_size=args.batch_size,
                                               shuffle=False)
    for _, mini_batch_seeds in enumerate(train_loader):
        gpu_sampler.sample(mini_batch_seeds)
    if i != 0:
        cache.pass_3(iterptr, iters, initial_cache_indices)
        torch.cuda.synchronize()
        return cache, initial_cache_indices.cpu(), time.time() - sampler_start
    else:
        torch.cuda.synchronize()
        return None, None, 0


def switch(cache, initial_cache_indices):
    cache.fill_cache(initial_cache_indices)
    del (initial_cache_indices)
    return cache


def trace_load(q, indices, sb):
    for i in indices:
        q.put((
            torch.load('./trace/' + args.exp_name + '/' + 'sb_' +
                       str(sb) + '_ids_' + str(i) + '.pth'),
            torch.load('./trace/' + args.exp_name + '/' + 'sb_' +
                       str(sb) + '_adjs_' + str(i) + '.pth'),
            torch.load('./trace/' + args.exp_name + '/' + 'sb_' +
                       str(sb) + '_update_' + str(i) + '.pth'),
        ))


def gather(gather_q, n_id, cache, batch_size):
    batch_inputs, _ = gather_ginex(feature_path, n_id, feature_dim, cache)
    batch_labels = labels[n_id[:batch_size]]
    gather_q.put((batch_inputs, batch_labels, n_id))


def delete_trace(i):
    n_id_filelist = glob.glob(
        './trace/' + args.exp_name + '/sb_' + str(i - 1) + '_ids_*')
    adjs_filelist = glob.glob(
        './trace/' + args.exp_name + '/sb_' + str(i - 1) + '_adjs_*')
    cache_filelist = glob.glob(
        './trace/' + args.exp_name + '/sb_' + str(i - 1) + '_update_*')

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

    q = list()
    loader = list()

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
                log.info(f"training batch_size: {batch_size}")
                n_id_q.put(n_id)
                adjs_q.put(adjs)
                in_indices_q.put(in_indices)

                in_positions_q.put(in_positions)
                out_indices_q.put(out_indices)

            # Gather
            log.info("begin gather")
            batch_inputs, _ = gather_ginex(
                feature_path, n_id, feature_dim, cache)
            log.info(
                f"loading batch input finish, input shape: {batch_inputs.shape}")
            batch_labels = labels[n_id[:batch_size]]
            log.info(f"loading label finish, labels: {batch_labels}")

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

        # Transfer
        batch_inputs_cuda = batch_inputs.to(device)
        batch_labels_cuda = batch_labels.to(device)
        adjs_host = adjs_q.get()
        adjs = [adj.to(device) for adj in adjs_host]

        # Forward
        n_id = n_id_q.get()
        n_id_cuda = n_id.to(device)
        out = model(batch_inputs_cuda, adjs)
        loss = F.nll_loss(out, batch_labels_cuda.long())

        # Backward
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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

    return total_loss, total_correct


def train_sample_gpu(epoch, gpu_sampler):
    model.train()
    neighbor_indice_time = 0
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
        cache, initial_cache_indices, sampler_batch_time = inspect_gpu(
            i, last=(i == num_sb), gpu_sampler=gpu_sampler, mode='train')
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
        total_loss, total_correct = execute(
            i, cache, pbar, total_loss, total_correct, last=(i == num_sb), mode='train')
        if args.verbose:
            tqdm.write('Step 3: Done')

        # Delete obsolete runtime files
        delete_trace(i)
        neighbor_indice_time += sampler_batch_time
        # TODO: refine it!
        gpu_sampler.inc_sb()

    pbar.close()

    loss = total_loss / num_iter
    approx_acc = total_correct / dataset.shuffled_train_idx.numel()
    log.info(f"epoch: {epoch}, evit time: {model.evit_time}, index select time: {model.index_select_time}, cache transfer time: {model.cache_transfer_time}, sampler indice \
            time: {neighbor_indice_time}")
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
        total_loss, total_correct = execute(
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
    if args.use_gpu:
        log.info("initialize gpu sampler..")
        gpu_sampler = get_sampler(edge_index)
        log.info(f"training parameter: {sizes}")
        for epoch in range(args.num_epochs):
            if args.verbose:
                tqdm.write('\n==============================')
                tqdm.write('Running Epoch {}...'.format(epoch))

            start = time.time()
            loss, acc = train_sample_gpu(epoch, gpu_sampler)
            end = time.time()
            log.info(
                f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
            log.info('Epoch time: {:.4f} ms'.format((end - start) * 1000))

            if epoch > 3 and not args.train_only and args.dataset in ["ogbn-papers100M", "ogbn-products", "igb-medium"]:
                val_loss, val_acc = inference(mode='valid')
                test_loss, test_acc = inference(mode='test')
                log.info('Valid loss: {0:.4f}, Valid acc: {1:.4f}, Test loss: {2:.4f}, Test acc: {3:.4f},'.format(
                    val_loss, val_acc, test_loss, test_acc))

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    final_test_acc = test_acc
            gpu_sampler.reset_sampler()
        if not args.train_only:
            log.info(f'Final Test acc: {final_test_acc:.4f}')
