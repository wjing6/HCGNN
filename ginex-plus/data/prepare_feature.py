import argparse
from ogb.nodeproppred import PygNodePropPredDataset
import scipy
import numpy as np
import json
import torch
import os


# Parse arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, default='ogbn-papers100M')
argparser.add_argument('--feature-cache-size', type=float, default=500000000)
argparser.add_argument('--static', dest='static', default=False, action='store_true')
args = argparser.parse_args()

# Download/load dataset
print('Loading dataset...')
root = '/data01/liuyibo/'
os.makedirs(root, exist_ok=True)
dataset = PygNodePropPredDataset(args.dataset, root)
dataset_path = os.path.join(root, args.dataset + '-ginex')
print('Done!')

# Construct sparse formats
def prepareRawData():
    print('Creating coo/csc/csr format of dataset...')
    num_nodes = dataset[0].num_nodes
    coo = dataset[0].edge_index.numpy()
    v = np.ones_like(coo[0])
    coo = scipy.sparse.coo_matrix((v, (coo[0], coo[1])), shape=(num_nodes, num_nodes))
    csc = coo.tocsc()
    csr = coo.tocsr()
    print('Done!')

    # Save csc-formatted dataset
    indptr = csc.indptr.astype(np.int64)
    indices = csc.indices.astype(np.int64)
    features = dataset[0].x
    labels = dataset[0].y

    os.makedirs(dataset_path, exist_ok=True)
    indptr_path = os.path.join(dataset_path, 'indptr.dat')
    indices_path = os.path.join(dataset_path, 'indices.dat')
    features_path = os.path.join(dataset_path, 'features.dat')
    labels_path = os.path.join(dataset_path, 'labels.dat')
    conf_path = os.path.join(dataset_path, 'conf.json')
    split_idx_path = os.path.join(dataset_path, 'split_idx.pth')

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

    print('Saving features...')
    features_mmap = np.memmap(features_path, mode='w+', shape=dataset[0].x.shape, dtype=np.float32)
    features_mmap[:] = features[:]
    features_mmap.flush()
    print('Done!')

    print('Saving labels...')
    labels = labels.type(torch.float32)
    labels_mmap = np.memmap(labels_path, mode='w+', shape=dataset[0].y.shape, dtype=np.float32)
    labels_mmap[:] = labels[:]
    labels_mmap.flush()
    print('Done!')

    print('Making conf file...')
    mmap_config = dict()
    mmap_config['num_nodes'] = int(dataset[0].num_nodes)
    mmap_config['indptr_shape'] = tuple(indptr.shape)
    mmap_config['indptr_dtype'] = str(indptr.dtype)
    mmap_config['indices_shape'] = tuple(indices.shape)
    mmap_config['indices_dtype'] = str(indices.dtype)
    mmap_config['indices_shape'] = tuple(indices.shape)
    mmap_config['indices_dtype'] = str(indices.dtype)
    mmap_config['indices_shape'] = tuple(indices.shape)
    mmap_config['indices_dtype'] = str(indices.dtype)
    mmap_config['features_shape'] = tuple(features_mmap.shape)
    mmap_config['features_dtype'] = str(features_mmap.dtype)
    mmap_config['labels_shape'] = tuple(labels_mmap.shape)
    mmap_config['labels_dtype'] = str(labels_mmap.dtype)
    mmap_config['num_classes'] = int(dataset.num_classes)
    json.dump(mmap_config, open(conf_path, 'w'))
    print('Done!')

    print('Saving split index...')
    torch.save(dataset.get_idx_split(), split_idx_path)
    print('Done!')

    # Calculate and save score for neighbor cache construction
    print('Calculating score for neighbor cache construction...')
    score_path = os.path.join(dataset_path, 'nc_score.pth')
    csc_indptr_tensor = torch.from_numpy(csc.indptr.astype(np.int64))
    csr_indptr_tensor = torch.from_numpy(csr.indptr.astype(np.int64))

    eps = 0.00000001
    in_num_neighbors = (csc_indptr_tensor[1:] - csc_indptr_tensor[:-1]) + eps
    out_num_neighbors = (csr_indptr_tensor[1:] - csr_indptr_tensor[:-1]) + eps
    score = out_num_neighbors / in_num_neighbors
    print('Done!')

    print('Saving score...')
    torch.save(score, score_path)
    print('Done!')
    
    
def prepareFeatureCache(cache_size, feature_dim):
    feature_size = feature_dim * 8  # feature type: (float)
    feature_num = int(cache_size // feature_size)
    print("=" * 10 + "Feature Cache Preprocess" + "=" * 10)
    num_nodes = dataset[0].num_nodes
    
    features_path = os.path.join(dataset_path, 'features.dat')
    features = np.memmap(features_path, mode='r', shape=dataset[0].x.shape, dtype=np.float32)
    
    coo = dataset[0].edge_index.numpy()
    v = np.ones_like(coo[0])
    coo = scipy.sparse.coo_matrix((v, (coo[0], coo[1])), shape=(num_nodes, num_nodes))
    csr = coo.tocsr()
    print('Done!')
    
    csr_indptr_tensor = torch.from_numpy(csr.indptr.astype(np.int64))
    eps = 0.00000001
    out_num_neighbors = (csr_indptr_tensor[1:] - csr_indptr_tensor[:-1]) + eps
    sorted_indices = out_num_neighbors.argsort(descending=True)
    
    print ("Feature: {:d}, Total num: {:d}".format(feature_num, num_nodes))
    if feature_num > num_nodes:
        save_feature_indice = sorted_indices
    else:
        save_feature_indice = sorted_indices[:feature_num]

    feature_cache = features[save_feature_indice]
    
    save_feature_indice = np.array(save_feature_indice, dtype=np.int64)
    feature_indice_path = os.path.join(dataset_path, 'feature_indice.dat')
    feature_cache_path = os.path.join(dataset_path, 'feature_cache.dat')
    print("=" * 10 + "Saving Feature Indice.." + "=" * 10)
    print("feature shape: ", save_feature_indice.shape)
    feature_cache_indice = np.memmap(feature_indice_path, mode='w+', shape=save_feature_indice.shape, dtype=np.int64)
    feature_cache_indice[:] = save_feature_indice[:]
    feature_cache_indice.flush()
    
    feature_cache_mmap = np.memmap(feature_cache_path, mode='w+', shape=feature_cache.shape, dtype=np.float32)
    feature_cache_mmap[:] = feature_cache[:]
    feature_cache_mmap.flush()
    
    print (save_feature_indice[:10])
    print (feature_cache_mmap[:10])
    print('Done!')


prepareRawData()
if args.static:
    prepareFeatureCache(args.feature_cache_size, 128)
