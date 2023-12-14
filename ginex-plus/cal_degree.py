import argparse
import torch
import numpy as np
import os
import scipy
from ogb.nodeproppred import PygNodePropPredDataset
import prepare_data_from_scratch
from random import sample
import json

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, default='ogbn-papers100M')
args = argparser.parse_args()

if args.dataset in ["igb-tiny", "igb-small", "igb-medium", "igb-large", "igb-full"]:
    import dgl
    from igb.dataloader import IGB260MDGLDataset
    args.path = '/data01/liuyibo/IGB260M/'
    args.dataset_size = 'medium'
    args.num_classes = 19
    args.in_memory = 0
    args.synthetic = 0


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
    csc_indptr = csc_mat.indptr.astype(np.int64)
    csr_indptr = csr_mat.indptr.astype(np.int64)
    dataset_folder = os.path.join(root, dataset)
    
    
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
        
    csr_indptr_shape = csr_indptr.shape
    csc_indptr_shape = csc_indptr.shape
    # return csr matrix indptr
    return csr_indptr, csr_indptr_shape, csc_indptr, csc_indptr_shape

def get_edge_index(dataset_name):
    if dataset_name in ["ogbn-products", 'ogbn-papers100M']:
        dataset = PygNodePropPredDataset(args.dataset, root)
        data = dataset[0]
        num_nodes = data.num_nodes
        edge_index = data.edge_index
    elif dataset_name in ["friendster", "twitter"]:
        dataset_folder = os.path.join(root, args.dataset)
    
        file_name = dataset_name + ".bin"
        dataset_path = os.path.join(dataset_folder, file_name)
        edge_index, num_nodes = prepare_data_from_scratch.load_edge_list_from_binary(dataset_path)
        edge_index = edge_index.t()
        print(edge_index.shape)
    elif dataset_name in ["igb-tiny", "igb-small", "igb-medium", "igb-large", "igb-full"]:
        dataset = IGB260MDGLDataset(args)
        graph = dataset[0]
        num_nodes = graph.num_nodes()
        edge_index = graph.edges()
    elif dataset_name in ["bytedata_caijing", "douyin_fengkong_guoqing_0709", 
                          "douyin_fengkong_sucheng_0813", "caijing_xiaowei_wangzhenchao", 
                          "bytedata_part"]:
        dataset_folder = os.path.join(root, args.dataset) # fengkong is not in /data01!
        
        file_name = "b.bin"
        dataset_path = os.path.join(dataset_folder, file_name)
        edge_index, num_nodes = prepare_data_from_scratch.load_edge_list_from_binary(dataset_path)
        edge_index = edge_index.t()
        print (edge_index.shape)
    else:
        print ("unsupported dataset!")
        exit(1)
    print("{dataset} num nodes is {:d}".format(num_nodes, dataset=args.dataset))

    # remove self-loop
    if dataset_name not in ["igb-tiny", "igb-small", "igb-medium", "igb-large", "igb-full"]:
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]
        print("remove self loop finish.. after remove, the edge number is {:d}".format(edge_index.shape[1]))
    return edge_index


def cal_degree_(node_id, csr_indptr, csr_indptr_shape, csc_indptr, csc_indptr_shape):
    '''
    calculate degree according the node_id, draw the distribution
    '''
    csr_indptr = torch.from_numpy(csr_indptr)
    print ("sample node: ", len(node_id), " indptr shape: ", csr_indptr_shape)
    node_id_plus = [id + 1 for id in node_id]
    degree = csr_indptr[node_id_plus] - csr_indptr[node_id]
    in_degree = csc_indptr[node_id_plus] - csc_indptr[node_id]
    all_degree = degree + in_degree
    print (all_degree)
    return degree


if __name__ == '__main__':
    edge_index = get_edge_index(args.dataset)
    csr_indptr, csr_indptr_shape, csc_indptr, csc_indptr_shape = prepare_topo(args.dataset, edge_index)
    dataset_folder = os.path.join(root, args.dataset)
    node_id_file = os.path.join(dataset_folder, args.dataset + '.nid')
    with open(node_id_file, "r") as fp:
        node_id = json.load(fp)
    degrees = cal_degree_(node_id, csr_indptr, csr_indptr_shape, csc_indptr, csc_indptr_shape)
    degree_path = os.path.join(dataset_folder, args.dataset + '.degree')
    print (degrees)
    torch.save(degrees, degree_path)
    