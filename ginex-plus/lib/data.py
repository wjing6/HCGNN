import os
import json
import numpy as np
import sys
import scipy
import torch
from lib.utils import *
from lib.cache import NeighborCache
import prepare_data_from_scratch
from ogb.nodeproppred import PygNodePropPredDataset
sys.path.append("../")
from log import log
from random import sample

def prepare_topo(root, dataset, edge_index, train_idx, feature_dim, need_score, num_classes = 1000):
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    num_nodes = max(np.max(src), np.max(dst)) + 1
    coo = edge_index.numpy()
    log.info(f"{dataset} nodes number: {num_nodes}")
    data = np.ones_like(coo[0])
    coo = scipy.sparse.coo_matrix((data, (coo[0], coo[1])), shape=(num_nodes, num_nodes))
    log.info("generate coo finish")
    csc_mat = coo.tocsc()
    log.info("csc matrix finish")
    indptr = csc_mat.indptr.astype(np.int64)
    indices = csc_mat.indices.astype(np.int64)
    dataset_folder = root
    os.makedirs(dataset_folder, exist_ok=True)
    indptr_path = os.path.join(dataset_folder, 'indptr.dat')
    indices_path = os.path.join(dataset_folder, 'indices.dat')
    features_path = os.path.join(dataset_folder, 'features.dat')
    labels_path = os.path.join(dataset_folder, 'labels.dat')
    conf_path = os.path.join(dataset_folder, 'conf.json')
    # split_idx_path = os.path.join(dataset_folder, 'split_idx.pth')
    log.info("sparse matrix transform finish")
    if dataset in ["ogbn-products", 'ogbn-papers100M']:
        dataset = PygNodePropPredDataset(dataset, root)
        features = dataset[0].x
        labels = dataset[0].y
        labels = labels.type(torch.float32)
        num_classes = dataset.num_classes
    else:
        if not os.path.exists(features_path) and not os.path.exists(labels_path):
            features = torch.randn((num_nodes, feature_dim), dtype=torch.float32)
            labels = torch.ones((num_nodes, 1), dtype=torch.float32)
        else:
            log.info("features and labels exist")

    if not os.path.exists(features_path):
        log.info('Saving features...')
        features_mmap = np.memmap(features_path, mode='w+', shape=features.shape, dtype=np.float32)
        features_mmap[:] = features[:]
        features_mmap.flush()
        log.info('Done!')

    if not os.path.exists(labels_path):
        log.info('Saving labels...')
        labels_mmap = np.memmap(labels_path, mode='w+', shape=labels.shape, dtype=np.float32)
        labels_mmap[:] = labels[:]
        labels_mmap.flush()
        log.info('Done!')

    if os.path.exists(indptr_path) and os.path.exists(indices_path):
        pass
    else:
        log.info('Saving indptr...')
        indptr_mmap = np.memmap(indptr_path, mode='w+', shape=indptr.shape, dtype=indptr.dtype)
        indptr_mmap[:] = indptr[:]
        indptr_mmap.flush()
        log.info('Done!')

        log.info('Saving indices...')
        indices_mmap = np.memmap(indices_path, mode='w+', shape=indices.shape, dtype=indices.dtype)
        indices_mmap[:] = indices[:]
        indices_mmap.flush()
        log.info('Done!')

    num_nodes = num_nodes.tolist() # numpy.int64 to 'int'
    log.info('Making conf file...')
    mmap_config = dict()
    mmap_config['num_nodes'] = int(num_nodes)
    mmap_config['indptr_shape'] = tuple(indptr.shape)
    mmap_config['indptr_dtype'] = str(indptr.dtype)
    mmap_config['indices_shape'] = tuple(indices.shape)
    mmap_config['indices_dtype'] = str(indices.dtype)
    mmap_config['features_shape'] = tuple([num_nodes, feature_dim])
    mmap_config['features_dtype'] = str('float32')
    mmap_config['labels_dtype'] = str('float32')
    mmap_config['labels_shape'] = tuple([num_nodes, 1])
    mmap_config['num_classes'] = int(num_classes)
    json.dump(mmap_config, open(conf_path, 'w'))
    log.info('Done!')

    if need_score:
        log.info('Calculating score for neighbor cache construction...')
        score_path = os.path.join(dataset_folder, 'nc_score.pth')

        csr_mat = coo.tocsr()
        log.info("csr matrix finish")
        
        csc_indptr_tensor = torch.from_numpy(csc_mat.indptr.astype(np.int64))
        csr_indptr_tensor = torch.from_numpy(csr_mat.indptr.astype(np.int64))
        log.info(f"{csc_indptr_tensor.shape}, {csr_indptr_tensor.shape}")
        if not os.path.exists(score_path):
            eps = 0.00000001
            in_num_neighbors = (csc_indptr_tensor[1:] - csc_indptr_tensor[:-1]) + eps
            out_num_neighbors = (csr_indptr_tensor[1:] - csr_indptr_tensor[:-1]) + eps
            score = out_num_neighbors / in_num_neighbors
            log.info('Saving score...')
            torch.save(score, score_path)
            log.info('Done!')        

    indptr_shape = indptr.shape
    return indptr_path, indptr_shape, indices_path, score_path


def get_edge_index(root, dataset_name, f_dim):
    if dataset_name in ["ogbn-products", 'ogbn-papers100M']:
        dataset_root = root.rpartition('/')[0]
        dataset = PygNodePropPredDataset(dataset_name, dataset_root)
        split_idx = dataset.get_idx_split()
        data = dataset[0]
        feature_dim = data.x.shape[1]
        num_nodes = data.num_nodes
        edge_index = data.edge_index
        train_idx = split_idx['train']
        dataset_path = os.path.join(dataset_root, dataset_name + '-ginex')
        split_idx_path = os.path.join(dataset_path, 'split_idx.pth')
        if not os.path.exists(split_idx_path):
            torch.save(split_idx, split_idx_path)
    elif dataset_name in ["friendster", "twitter"]:
        feature_dim = f_dim
        binary_path = os.path.join(root, dataset_name + ".bin")
        if not os.path.exists(binary_path):
            file_name = dataset_name + ".csv"
            dataset_path = os.path.join(root, file_name)
            log.info(f"data path: {dataset_path}")
            edge_index, num_nodes = prepare_data_from_scratch.load_edge_list(
                dataset_path, "#", True)
        else:
            edge_index, num_nodes = prepare_data_from_scratch.load_edge_list_from_binary(binary_path)
        edge_index = edge_index.t()
        log.info(f"edge_index.shape: {edge_index.shape}")
        train_idx_path = os.path.join(root, "train_idx.pth")
        if os.path.exists(train_idx_path):
            train_idx = torch.load(train_idx_path)
        else:
            train_idx = sample(range(num_nodes), int(0.1 * num_nodes))
            train_idx = torch.tensor(train_idx)
            torch.save(train_idx, train_idx_path)
    elif dataset_name in ["bytedata_caijing", "douyin_fengkong_guoqing_0709", 
                          "douyin_fengkong_sucheng_0813", "caijing_xiaowei_wangzhenchao", 
                          "bytedata_part"]:
        feature_dim = f_dim
        dataset_folder = os.path.join(root, dataset_name)
        if not os.path.exists("b.bin"):
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
        train_idx_path = os.path.join(dataset_folder, "train_idx.pth")
        if os.path.exists(train_idx_path):
            train_idx = torch.load(train_idx_path)
        else:
            train_idx = sample(range(num_nodes), int(0.1 * num_nodes))
            train_idx = torch.tensor(train_idx)
            torch.save(train_idx, train_idx_path)
    else:
        log.error("unsupported dataset!")
        sys.exit(0)
    log.info(f"{dataset_name} feature dimension is {feature_dim}, num nodes is {num_nodes}")
    # remove self-loop
    if dataset_name not in ["igb-tiny", "igb-small", "igb-medium", "igb-large", "igb-full"]:
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]
        log.info(f"remove self loop finish.. after remove, the edge number is {format(edge_index.shape[1])}")
    
    return edge_index, num_nodes, feature_dim, train_idx

class GinexDataset():
    def __init__(self, path, dataset=None, split_idx_path=None, score_path=None, need_score=False):
        self.root_path = path
        self.dataset = dataset
        self.feature_dim = 256 
        self.need_score=need_score # need_score 代表是否需要处理 score - GPU
        # 这里的 feature_dim 用来在未指定feature的数据集上 prepare 使用
        # 指定 feature_dim 的数据集或者已经存在conf.json的数据集, feature_dim 会在后面更新
        self.indptr_path = os.path.join(self.root_path, 'indptr.dat')
        self.indices_path = os.path.join(self.root_path, 'indices.dat')
        self.features_path = os.path.join(self.root_path, 'features.dat')
        self.labels_path = os.path.join(self.root_path, 'labels.dat')
        conf_path = os.path.join(self.root_path, 'conf.json')
        score_path = os.path.join(self.root_path, 'nc_score.pth')
        if not os.path.exists(conf_path):
            log.info(f"{conf_path}")            
            self.prepare_dataset()
        self.conf = json.load(open(conf_path, 'r'))
        self.feature_dim = self.conf['features_shape'][1]
        self.edge_index, _, _, _ = get_edge_index(self.root_path, self.dataset, self.feature_dim)
        

        if os.path.exists(split_idx_path):
            split_idx = torch.load(split_idx_path)
            self.train_idx = split_idx['train']
            self.val_idx = split_idx['valid']
            self.test_idx = split_idx['test']
        else:
            train_idx_path = os.path.join(self.root_path, 'train_idx.pth')
            self.train_idx = torch.load(train_idx_path)
        
        self.score_path = score_path

        self.num_nodes = self.conf['num_nodes']
        self.num_classes = self.conf['num_classes']


    # Return indptr & indices
    def get_adj_mat(self):
        indptr = np.fromfile(self.indptr_path, dtype=self.conf['indptr_dtype']).reshape(tuple(self.conf['indptr_shape']))
        indices = np.memmap(self.indices_path, mode='r', shape=tuple(self.conf['indices_shape']), dtype=self.conf['indices_dtype'])
        indptr = torch.from_numpy(indptr)
        indices = torch.from_numpy(indices)
        return indptr, indices


    def get_col(self):
        indices = np.memmap(self.indices_path, mode='r', shape=tuple(self.conf['indices_shape']), dtype=self.conf['indices_dtype'])
        indices = torch.from_numpy(indices)
        return indices


    def get_rowptr_mt(self):
        indptr_size = self.conf['indptr_shape'][0]
        indptr = mt_load(self.indptr_path, indptr_size)
        return indptr


    def get_labels(self):
        labels = torch.from_numpy(np.fromfile(self.labels_path, dtype=self.conf['labels_dtype'], count=self.num_nodes).reshape(tuple([self.conf['labels_shape'][0]])))
        return labels


    def get_labels_mt(self):
        labels_size = self.conf['labels_shape'][0]
        labels = mt_load_float(self.labels_path, labels_size)
        return labels


    def make_new_shuffled_train_idx(self):
        self.shuffled_train_idx = self.train_idx[torch.randperm(self.train_idx.numel())]


    def get_mmapped_features(self):
        features_shape = self.conf['features_shape']
        features = np.memmap(self.features_path, mode='r', shape=tuple(features_shape), dtype=self.conf['features_dtype'])
        features = torch.from_numpy(features)
        return features


    def get_score(self):
        return torch.load(self.score_path)

    def prepare_dataset(self):
        edge_index, _, feature_dim, train_idx = get_edge_index(self.root_path, self.dataset, self.feature_dim)
        self.edge_index = edge_index
        self.train_idx = train_idx
        prepare_topo(self.root_path, self.dataset, edge_index, train_idx, feature_dim, self.need_score)
        log.info("prepare dataset finished")
    
    def get_edge_and_train(self):
        # TODO: 统一 quiver 和 ginex 的调用接口, 减少 edge_index 和 indptr 的冗余存储
        return self.edge_index, self.train_idx


    def save_neighbor_cache(self, neigh_cache_size = None):
        log.info('Creating neighbor cache...')
        score = self.get_score()
        rowptr, col = self.get_adj_mat()
        num_nodes = self.num_nodes
        if neigh_cache_size is not None:
            # TODO: NeighborCache 同时支持 prop 和 size 参数
            neighbor_cache = NeighborCache(neigh_cache_size, score, rowptr, self.indices_path, num_nodes)
        else:
            # cache all the neighbor
            neighbor_cache = NeighborCache(0, score, rowptr, self.indices_path, num_nodes, 1)
        del(score)

        log.info('Saving neighbor cache...')
        if neigh_cache_size is not None:
            cache_filename = str(self.root_path) + '/nc_size_' + str(neigh_cache_size)
            neighbor_cache.save(neighbor_cache.cache.numpy(), cache_filename)
            cache_tbl_filename = str(self.root_path) + '/nctbl_size_' + str(neigh_cache_size)
            neighbor_cache.save(neighbor_cache.address_table.numpy(), cache_tbl_filename)
        else:
            cache_filename = str(self.root_path) + '/nc_all'
            neighbor_cache.save(neighbor_cache.cache.numpy(), cache_filename)
            cache_tbl_filename = str(self.root_path) + '/nctbl_all'
            neighbor_cache.save(neighbor_cache.address_table.numpy(), cache_tbl_filename)
        log.info('Saving neighbor cache done!')
