import os
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
from ogb.nodeproppred import PygNodePropPredDataset
# from torch_geometric.datasets import Reddit
import quiver
import time
import numpy as np
from collections import OrderedDict
from lib.belady_cache import beladyCache
from lib.neighbor_sampler import GinexNeighborSampler
from lib.cache import NeighborCache
import argparse
from random import sample
import sys
import prepare_data_from_scratch
from collections import Counter
import pandas as pd
import scipy
import random
sys.path.append("../")

from lib.classical_cache import LRUCache, FIFO
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


def parse_size(sz) -> int:
    if isinstance(sz, int):
        return sz
    elif isinstance(sz, float):
        return int(sz)
    elif isinstance(sz, str):
        for suf, u in sorted(UNITS.items()):
            if sz.upper().endswith(suf):
                return int(float(sz[:-len(suf)]) * u)
    raise Exception("invalid size: {}".format(sz))


batch_for_dataset = {"ogbn-papers100M": 4096,
                     "ogbn-products": 1024,
                     "friendster": 4096,
                     "igb-medium": 4096,
                     "twitter": 4096,
                     "bytedata_caijing": 4096,
                     "bytedata_part":4096,
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
argparser.add_argument('--cache-policy', type=str, default='belady')
argparser.add_argument('--mode', type=str, choices=['train', 'test-neighbor'], default='train')
argparser.add_argument('--feature-dim', type=int, default=256)
argparser.add_argument('--exist-binary', dest='exist_binary',
                       default=False, action='store_true')
argparser.add_argument('--neigh-prop', type=float, default=0.1)
argparser.add_argument('--fifo-length', type=int, default=500000)
argparser.add_argument('--epoch', type=int, default=1)
args = argparser.parse_args()

# dataset argument
if args.dataset in ["igb-tiny", "igb-small", "igb-medium", "igb-large", "igb-full"]:
    import dgl
    from igb.dataloader import IGB260MDGLDataset
    argparser.add_argument('--path', type=str, default='/data01/liuyibo/IGB260M/',
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

args = argparser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
os.environ['GINEX_NUM_THREADS'] = str(128)
print(str(args.gpu), torch.cuda.device_count())



def get_static_cache_by_pre_sample(capacity, train_idx, batch_size, sampler):
    cache_idx = set()  # for fast search
    train_loader = torch.utils.data.DataLoader(train_idx,
                                               batch_size=batch_size,
                                               shuffle=False)
    for _, mini_batch_seed in enumerate(train_loader):
        n_id, _, _ = sampler.sample(mini_batch_seed)
        if sampler.mode != 'CPU':
            n_id = n_id.to('cpu')
        n_id = n_id.tolist()
        if len(cache_idx) <= capacity:
            for id in n_id:
                if id not in cache_idx:
                    cache_idx.add(id)
                if len(cache_idx) > capacity:
                    break
    return cache_idx


root = "/data01/liuyibo/"

def prepare_topo(dataset, edge_index):
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    num_nodes = max(np.max(src), np.max(dst)) + 1
    coo = edge_index.numpy()
    print (num_nodes)
    data = np.ones_like(coo[0])
    coo = scipy.sparse.coo_matrix((data, (coo[0], coo[1])), shape=(num_nodes, num_nodes))
    csc_mat = coo.tocsc()
    csr_mat = coo.tocsr()
    indptr = csc_mat.indptr.astype(np.int64)
    indices = csc_mat.indices.astype(np.int64)
    dataset_folder = os.path.join(root, dataset)
    
    indptr_path = os.path.join(dataset_folder, 'indptr.dat')
    indices_path = os.path.join(dataset_folder, 'indice.dat')
    if os.path.exists(indptr_path) and os.path.exists(indices_path):
        pass
    else:
        print('Saving indptr...')
        indptr_mmap = np.memmap(indptr_path, mode='w+', shape=indptr.shape, dtype=indptr.dtype)
        indptr_mmap[:] = indptr[:]
        indptr_mmap.flush()
        print('Done!')

        print('Saving indices...')
        indices_mmap = np.memmap(indices_path, mode='w+', shape=indices.shape, dtype=indices.dtype)
        indices_mmap[:] = indices[:]
        indices_mmap.flush()
        print('Done!')
    
    print('Calculating score for neighbor cache construction...')
    score_path = os.path.join(dataset_folder, 'nc_score.pth')
    csc_indptr_tensor = torch.from_numpy(csc_mat.indptr.astype(np.int64))
    csr_indptr_tensor = torch.from_numpy(csr_mat.indptr.astype(np.int64))
    print (csc_indptr_tensor.shape, csr_indptr_tensor.shape)
    if not os.path.exists(score_path):
        eps = 0.00000001
        in_num_neighbors = (csc_indptr_tensor[1:] - csc_indptr_tensor[:-1]) + eps
        out_num_neighbors = (csr_indptr_tensor[1:] - csr_indptr_tensor[:-1]) + eps
        score = out_num_neighbors / in_num_neighbors
        
        print('Saving score...')
        torch.save(score, score_path)
        print('Done!')
        
    indptr_shape = indptr.shape
    return indptr_path, indptr_shape, indices_path, score_path
    
    
    


def get_dataset_and_sampler(dataset_name):
    if dataset_name in ["ogbn-products", 'ogbn-papers100M']:
        dataset = PygNodePropPredDataset(args.dataset, root)
        split_idx = dataset.get_idx_split()
        data = dataset[0]
        feature_dim = data.x.shape[1]
        num_nodes = data.num_nodes
        edge_index = data.edge_index
        train_idx = split_idx['train']

    elif dataset_name in ["friendster", "twitter"]:
        dataset_folder = os.path.join(root, args.dataset)
        feature_dim = args.feature_dim
        if not args.exist_binary:
            file_name = dataset_name + ".csv"
            dataset_path = os.path.join(dataset_folder, file_name)
            edge_index, num_nodes = prepare_data_from_scratch.load_edge_list(
                dataset_path, "#", True)
        else:
            file_name = dataset_name + ".bin"
            dataset_path = os.path.join(dataset_folder, file_name)
            edge_index, num_nodes = prepare_data_from_scratch.load_edge_list_from_binary(dataset_path)
        edge_index = edge_index.t()
        print(edge_index.shape)
        train_idx_path = os.path.join(dataset_folder, "train_idx.npy")
        if os.path.exists(train_idx_path):
            train_idx = np.load(train_idx_path).tolist()
        else:
            train_idx = sample(range(num_nodes), int(0.1 * num_nodes))
            np.save(train_idx_path, np.array(train_idx))
    elif dataset_name in ["igb-tiny", "igb-small", "igb-medium", "igb-large", "igb-full"]:
        dataset = IGB260MDGLDataset(args)
        graph = dataset[0]
        feature_dim = graph.ndata['feat'].shape[1]
        train_idx = torch.nonzero(graph.ndata['train_mask'], as_tuple=True)[0]
        num_nodes = graph.num_nodes()
        edge_index = graph.edges()
    elif dataset_name in ["bytedata_caijing", "douyin_fengkong_guoqing_0709", 
                          "douyin_fengkong_sucheng_0813", "caijing_xiaowei_wangzhenchao", 
                          "bytedata_part"]:
        feature_dim = args.feature_dim
        dataset_folder = os.path.join(root, args.dataset) # fengkong is not in /data01!
        if not args.exist_binary:
            edge_table_folder = os.path.join(dataset_folder, "edge_tables")
            if dataset_name in ["bytedata_caijing"]:
                edge_index, num_nodes = prepare_data_from_scratch.load_from_part_file(edge_table_folder, True, True)
            else:
                edge_index, num_nodes = prepare_data_from_scratch.load_from_part_file(edge_table_folder, True, False)
        else:
            file_name = "b.bin"
            dataset_path = os.path.join(dataset_folder, file_name)
            edge_index, num_nodes = prepare_data_from_scratch.load_edge_list_from_binary(dataset_path)
        edge_index = edge_index.t()
        print (edge_index.shape)
        train_idx_path = os.path.join(dataset_folder, "train_idx.npy")
        if os.path.exists(train_idx_path):
            train_idx = np.load(train_idx_path).tolist()
        else:
            train_idx = sample(range(num_nodes), int(0.1 * num_nodes))
            np.save(train_idx_path, np.array(train_idx))
    else:
        print ("unsupported dataset!")
        exit(1)
    print("{dataset} feature dimension is {:d}, num nodes is {:d}".format(
        feature_dim, num_nodes, dataset=args.dataset))
    print (train_idx[0:5])

    # remove self-loop
    if dataset_name not in ["igb-tiny", "igb-small", "igb-medium", "igb-large", "igb-full"]:
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]
        print("remove self loop finish.. after remove, the edge number is {:d}".format(edge_index.shape[1]))
    trans_start = time.time()
    if args.mode == 'train':
        csr_topo = quiver.CSRTopo(edge_index)
        print("csr topo cost: {:6f}s ".format(time.time() - trans_start))
        neigh_sampler = quiver.pyg.GraphSageSampler(
            csr_topo, sizes=[15, 10, 5], device=args.gpu, mode='UVA')
        print ("indptr size: ", csr_topo.indptr.size())
        print ("indptr: ", csr_topo.indptr[-5:])
        print ("indice: ", csr_topo.indices[-5:])
    else:
        indptr_path, indptr_shape, indices_path, score_path = prepare_topo(args.dataset, edge_index)
        indptr = np.fromfile(indptr_path, dtype=np.int64).reshape(tuple(indptr_shape))
        indptr = torch.from_numpy(indptr)
        score = torch.load(score_path)
        '''
        Now use prop for test, so we can set 'cache_size' as 0
        '''
        neighbor_cache = NeighborCache(0, score, indptr, indices_path, num_nodes, args.neigh_prop)
        neigh_sampler = GinexNeighborSampler(indptr, indices_path, node_idx=torch.from_numpy(np.array(train_idx)),
                                       sizes=[15, 10, 5], num_nodes = num_nodes,
                                       cache_data = neighbor_cache.cache, address_table = neighbor_cache.address_table,
                                       batch_size=batch_for_dataset[args.dataset],
                                       shuffle=False)
    print("sampler init cost: {:6f}s ".format(time.time() - trans_start))
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return feature_dim, train_idx, num_nodes, neigh_sampler


feature_dim, train_idx, num_nodes, neigh_sampler = get_dataset_and_sampler(
    args.dataset)
feature_size = feature_dim * 4  # 4 = sizeof(float)


def test_belady_cache_hit_rate(epoch, num_sb, batch_size=batch_for_dataset[args.dataset], sb_size=args.sb_size, repeat=False):
    # num_cache_entries = int(args.feature_cache_size // feature_size)
    global train_idx
    num_cache_entries = int(args.prop * num_nodes)
    print("In dataset {dataset}, the cache size is {:6f} MB".format(
        float(num_cache_entries) * feature_size / UNITS["MB"], dataset=args.dataset))
    for i in range(epoch):
        train_idx = np.random.permutation(train_idx)
        pbar = tqdm(total=len(train_idx))
        pbar.set_description(f'Epoch {i:02d}')
        hit = 0
        total_obj = 0
        for cur_sb in (range(num_sb)):
            start_idx = cur_sb * batch_size * sb_size
            end_idx = min((cur_sb + 1) * batch_size * sb_size, len(train_idx))
            subtrain_loader = torch.utils.data.DataLoader(train_idx[start_idx:end_idx],
                                                          batch_size=batch_size,
                                                          shuffle=False)
            super_nid = []
            appearance = np.zeros(num_nodes)
            for _, mini_batch_seeds in enumerate(subtrain_loader):
                n_id, _, _ = neigh_sampler.sample(mini_batch_seeds)
                if neigh_sampler.mode != 'CPU':
                    n_id = n_id.to('cpu')
                super_nid.append(n_id)
                appearance[n_id.tolist()] += 1
            appearance_hit = np.argwhere(appearance > 0).tolist()
            if cur_sb == 0:
                # 代表 super-batch 覆盖到的节点
                print ("touch rate: {:.4f}%".format(float(len(appearance_hit)) / num_nodes * 100))
            belady_cache = beladyCache(
                len(super_nid), super_nid, num_nodes, num_cache_entries, pbar)
            belady_cache.simulate()
            hit_inc, total_inc = belady_cache.get_hit_and_total_obj()
            hit += hit_inc
            total_obj += total_inc
            pbar.update(end_idx - start_idx + 1)
            
            if repeat: # test stability when sorting randomly
                for i in range(5):
                    random.shuffle(super_nid)
                    print (super_nid[0:2])
                    belady_cache = beladyCache(len(super_nid), super_nid, num_nodes, num_cache_entries, pbar)
                    belady_cache.simulate()
        pbar.write("In global iteration, the cache hit rate is {:4f}% ".format(float(hit) / total_obj * 100))


def test_LRU_cache_hit_rate(epoch, batch_size):  # LRU don't need pre-sample
    global train_idx
    num_cache_entries = int(args.prop * num_nodes)
    print("In dataset {dataset}, cache entries: {:d}, the cache size is {:6f} MB".format(num_cache_entries,
        float(num_cache_entries) * feature_size / UNITS["MB"], dataset=args.dataset))
    num_batch = int(len(train_idx) // batch_size) + 1
    for i in range(epoch):
        train_idx = np.random.permutation(train_idx)
        pbar = tqdm(total=len(train_idx))
        pbar.set_description(f'Epoch {i:02d}')
        train_loader = torch.utils.data.DataLoader(train_idx,
                                                   batch_size=batch_size,
                                                   shuffle=False)
        LRU_cache = LRUCache(num_cache_entries)
        cache_hit_rate = 0.0
        for _, mini_batch_seeds in enumerate(train_loader):
            n_id, _, _ = neigh_sampler.sample(mini_batch_seeds)
            if neigh_sampler.mode != 'CPU':
                n_id = n_id.to('cpu')
            if not LRU_cache.is_full():
                for id in n_id:
                    LRU_cache.set(id.item(), 1)
                    if LRU_cache.is_full():
                        break
        pbar.write("LRU cache warm-up complete! ")
        del (train_loader)
        train_loader = torch.utils.data.DataLoader(train_idx,
                                                   batch_size=batch_size,
                                                   shuffle=False)
        for _, mini_batch_seeds in enumerate(train_loader):
            n_id, _, _ = neigh_sampler.sample(mini_batch_seeds)
            item_in_cache = list(LRU_cache.cache.keys())
            curr_hit_rate = float(len(set(item_in_cache) & set(
                n_id.tolist()))) / len(n_id.tolist()) * 100
            for id in n_id:
                LRU_cache.get(id.item())  # simulate the access pattern
            cache_hit_rate += curr_hit_rate
            one_hit_ratio = 0
            replace_hit = 0
            for id in n_id:
                item = LRU_cache.set(id.item(), 1)
                if item != None:
                    replace_hit += 1
                    if item[1] == 1:
                        one_hit_ratio += 1
            print(curr_hit_rate)
            if replace_hit != 0:
                print("one hit rate: {:5f}".format(float(one_hit_ratio) / replace_hit * 100))
            pbar.update(len(mini_batch_seeds.tolist()))
        pbar.write("In iteration {:d}, the cache hit rate is {:6f} %".format(
            i, cache_hit_rate / num_batch))


def test_stale_embedding_cache(epoch, batch_size, k_hop = 2):
    global train_idx
    num_feature_cache_entries = int(args.prop * num_nodes)
    num_embedding_cache = dict()
    for i in range(k_hop - 1):
        tag = 'layer_' + str(i)
        # layer_0 means the second from the bottom
        #       layer-(k_hop-1) embedding cache
        #           layer-1 embedding cache
        #           layer-0 embedding cache
        #               feature cache
        num_embedding_cache[tag] = FIFO(num_nodes, tag, 0.01)
    
    print("In dataset {dataset}, the cache size is {:6f} MB".format(
        float(num_feature_cache_entries) * feature_size / UNITS["MB"], dataset=args.dataset))
    num_batch = int(len(train_idx) // batch_size) + 1
    for i in range(epoch):
        train_idx = np.random.permutation(train_idx)
        pbar = tqdm(total=len(train_idx))
        pbar.set_description(f'Epoch {i:02d}')
        cache_hit_rate = 0.0
        train_loader = torch.utils.data.DataLoader(train_idx,
                                                   batch_size=batch_size,
                                                   shuffle=False)
        for _, mini_batch_seeds in enumerate(train_loader):
            nid, _, adjs = neigh_sampler.sample(mini_batch_seeds)
            if (neigh_sampler.mode != 'CPU'):
                n_id = n_id.to("cpu")
                adjs = [adj.to("cpu") for adj in adjs]
            for i, (_, _, size) in enumerate(adjs):
                x_target = n_id[:size[1]]
                print (f'layer-{i}, {x_target}')
                # target node always placed at head
                if i != len(adjs) - 1:
                    # not the top node
                    tag = 'layer_' + str(i)
                    num_hit = 0
                    for target_node in x_target:
                        if num_embedding_cache[tag].get(target_node):
                            # hit
                            num_hit += 1
                    print (f'num-hit: {num_hit}')
            
            pbar.update(len(mini_batch_seeds.tolist()))

        pbar.write("In iteration {:d}, the cache hit rate is {:6f} %".format(
            i, cache_hit_rate / num_batch))


def test_pre_sample_static_cache(epoch, batch_size):
    global train_idx
    num_cache_entries = int(args.prop * num_nodes)
    print("In dataset {dataset}, cache entries: {:d}, the cache size is {:6f} MB".format( num_cache_entries,
        float(num_cache_entries) * feature_size / UNITS["MB"], dataset=args.dataset))
    # num_batch = int((len(train_idx) - 1) // batch_size) + 1
    for e in range(epoch):
        train_idx = np.random.permutation(train_idx)
        pbar = tqdm(total=len(train_idx))
        pbar.set_description(f'Epoch {e:02d}')
        pre_sample_loader = torch.utils.data.DataLoader(train_idx,
                                                        batch_size=batch_size,
                                                        shuffle=False)
        access_stat = np.zeros(num_nodes)
        access_list = list()
        frq = torch.zeros(num_nodes, dtype=torch.int64, device='cpu')
        for iter, mini_seeds in enumerate(pre_sample_loader):
            print (mini_seeds.tolist(), file=f) 
            n_id, _, _ = neigh_sampler.sample(mini_seeds)
            if neigh_sampler.mode != 'CPU':
                n_id = n_id.to('cpu')
            n_id_list = n_id.tolist()
            if iter == 0:
                print ("touch rate: {:.4f}%".format(float(len(n_id)) / num_nodes * 100))
            access_stat[n_id_list] += 1  # speed!
            frq[n_id] += 1
            access_list.append(n_id)
            pbar.update(batch_size)
        
        # for n_id in access_list:
        #     n_id = n_id.cpu()
        print ("Pass : making two key data structures (iterptr & iters)...")
        msb = (torch.tensor([1], dtype=torch.int64) << 63).cpu()
        cumsum = frq.cumsum(dim=0)
        iterptr = torch.cat([torch.tensor([0,0], device='cpu'), cumsum[:-1]]); del(cumsum)
        frq_sum = frq.sum(); del(frq)

        iters = torch.zeros(frq_sum+1, dtype=torch.int64, device='cpu')
        iters[-1] = sys.maxsize

        for i, n_id in enumerate(access_list):
            n_id_cuda = n_id.cpu() 
            tmp = iterptr[n_id_cuda+1]
            iters[tmp] = i + 1; del(tmp)
            iterptr[n_id_cuda+1] += 1; del(n_id_cuda)
        iters[iterptr[1:]] |= msb
        iterptr = iterptr[:-1]
        iterptr[0] = 0
        print (iterptr[0:5])
        print ("Pass: making finish ...")
        n_id_list = []
        access_time = []
        print ("total access number: {:d}".format(frq_sum))
        access_size = [feature_size] * frq_sum
        next_access_time = []
        for iter, n_id in enumerate(access_list):
            n_id_list.extend(n_id.tolist())
            cur_time = [(iter + 1) * 5] * len(n_id.tolist())
            n_id_cuda = n_id.cpu()
            access_time.extend(cur_time)
            tmp = iterptr[n_id_cuda] + 1
            next_access = iters[tmp]
            next_access[next_access > 0] *= 5
            next_access[next_access < 0] = sys.maxsize
            next_access = next_access.cpu().tolist()
            next_access_time.extend(next_access)
            # update iterptr and iter
            iterptr[n_id_cuda] += 1
            last_access = n_id_cuda[(iters[iterptr[n_id_cuda]] < 0)]
            iterptr[last_access] = iters.numel()-1; del(last_access); del(n_id_cuda)
            pbar.update(batch_size)
        
        # output = root + args.dataset + '.csv'
        # dataframe = pd.DataFrame({'time': access_time,'object': n_id_list, 'size': access_size, 'next_access': next_access_time})
        # # save as csv, in order to use libCacheSim
        # dataframe.to_csv(output, index=False, sep=',', header=False)
            
        # # no_one_hit_indice = np.argwhere(access_stat >= 2).flatten()
        total_indice = np.argwhere(access_stat > 0).flatten()
        # print ("one hit rate: {:4f}%".format(float(len(total_indice) - len(no_one_hit_indice)) / len(total_indice) * 100))
        # result = Counter(access_stat[total_indice].tolist())
        # result = dict(result)
        # with open(args.dataset + '_access.csv', 'w', newline='') as f:
        #     writer = csv.writer(f)
        #     for row in result.items():
        #         writer.writerow(row)
        access_indice = np.argsort(-access_stat)
        print("= Top 5 hot: ", access_indice[0:5], " access number: ", access_stat[access_indice[0:5]])
        pre_sample_cache = set(access_indice[0:num_cache_entries].tolist())

        print("=" * 5 + "pre sample to get hotness finish, the cache rate is {:4f}%".format(float(len(total_indice)) / num_nodes * 100) + "=" * 5)

        # train_loader = torch.utils.data.DataLoader(train_idx,
        #                                            batch_size=batch_size,
        #                                            shuffle=False)
        cache_hit = 0
        total_obj = 0
        for _, n_id in enumerate(access_list):
            # n_id, _, _ = neigh_sampler.sample(mini_seeds)
            # if neigh_sampler.mode != 'CPU':
            #     n_id = n_id.to('cpu')
            n_id = n_id.tolist()
            cache_hit += len(pre_sample_cache & set(n_id))
            total_obj += len(n_id)
        cache_hit -= len(pre_sample_cache)
        pbar.write("In iteration {:d}, the cache hit rate is {:6f} %".format(e, float(cache_hit) / total_obj * 100))


def test_s3fifo_cache_hit_rate(epoch, batch_size):
    global train_idx
    num_cache_entries = int(args.prop * num_nodes)
    print("In dataset {dataset}, cache entries: {:d}, the cache size is {:6f} MB".format(num_cache_entries,
        float(num_cache_entries) * feature_size / UNITS["MB"], dataset=args.dataset))
    num_batch = int(len(train_idx) // batch_size) + 1
    for e in range(epoch):
        train_idx = np.random.permutation(train_idx)
        pbar = tqdm(total=len(train_idx))
        pbar.set_description(f'Epoch {e:02d}')
        
        s3_cache = s3FIFO(num_cache_entries, pbar, threshold=2, fifo_size_ratio=0.06, fifo_length=args.fifo_length, use_ratio=True)
        train_loader = torch.utils.data.DataLoader(train_idx,
                                                   batch_size=batch_size,
                                                   shuffle=False)
        
        cache_hit_rate = 0.0
        for _, mini_seeds in enumerate(train_loader):
            n_id, _, _ = neigh_sampler.sample(mini_seeds)
            if neigh_sampler.mode != 'CPU':
                n_id = n_id.to('cpu')
            n_id = n_id.tolist()
            num_cache_hit = 0
            for id in n_id:
                if s3_cache.get(id):
                    num_cache_hit += 1
            cache_hit_rate += float(num_cache_hit) / len(n_id) * 100
            # print ("fifo cache hit: {:4f}".format(float(num_cache_hit) / len(n_id) * 100))
            pbar.update(batch_size)
        pbar.write("In iteration {:d}, the cache hit rate is {:6f} %".format(
            e, cache_hit_rate / num_batch))


def test_prop_neigh_cache():
    print ("==== ENTER TESTING NEIGHBOR CACHE ====")
    pbar = tqdm(total=len(train_idx))
    pbar.set_description(f'TESTING -- ')
    sample_start = time.time()
    for iter, out in enumerate(neigh_sampler):
        if iter == 0:
            print(out)
        pbar.update(out[0])
    
    print ("In cache {:4f}% nodes, the sample time is {:4f}".format(args.neigh_prop, time.time() - sample_start))

if __name__ == '__main__':
    sb_size = args.sb_size
    batch_size = batch_for_dataset[args.dataset]
    num_sb = int((len(train_idx) - 1) / (sb_size * batch_size)) + 1
    # test_sampler(10, num_sb, batch_size, sb_size)
    if args.mode == 'test-neighbor':
        test_prop_neigh_cache()
    else:
        if args.cache_policy == 'belady':
            test_belady_cache_hit_rate(args.epoch, num_sb, repeat=False) # now test sorting
        elif args.cache_policy == 'LRU':
            test_LRU_cache_hit_rate(args.epoch, batch_size)
        elif args.cache_policy == 'embedding_cache':
            test_stale_embedding_cache(args.epoch, batch_size)
        elif args.cache_policy == 'static_mode':  # pre-sample one epoch
            test_pre_sample_static_cache(args.epoch, batch_size)
        elif args.cache_policy == 'fifo':
            test_s3fifo_cache_hit_rate(args.epoch, batch_size)
        else:
            print("Unsupported mode.")
        # debug_pre_sample()
