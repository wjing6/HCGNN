from lib.cpp_extension.wrapper import *
import torch

def tensor_free(t):
    free.tensor_free(t)


def gather_ginex(feature_file, idx, feature_dim, cache):
    return gather.gather_ginex(feature_file, idx, feature_dim, cache.cache, cache.address_table)

def gather_ginex_async(feature_file, idx, feature_dim, cache):
    return gather.gather_ginex_async(feature_file, idx, feature_dim, cache.cache, cache.address_table)


def uvm_alloc_indice(num_nodes):
    return gather.alloc_uvm_indice(num_nodes)


def gather_mmap(features, idx):
    return gather.gather_mmap(features, idx, features.shape[1])

def get_size(features):
    if (features.dtype == torch.int16) or (features.dtype == torch.float16):
        element_size = 2
    elif (features.dtype == torch.int32) or (features.dtype == torch.float32):
        element_size = 4
    elif (features.dtype == torch.int64) or (features.dtype == torch.float64):
        element_size = 8
    elif features.dtype == torch.int8:
        element_size = 1
    else:
        return -1
    return gather.get_size_in_bytes(features, element_size)


def load_float32(path, size):
    return mt_load.load_float32(path, size)


def load_int64(path, size):
    return mt_load.load_int64(path, size)


def cache_update(cache, batch_inputs, in_indices, in_positions, out_indices):
    update.cache_update(cache.cache, cache.address_table, batch_inputs, in_indices, in_positions, out_indices, cache.cache.shape[1])


def fill_neighbor_cache(cache, rowptr, col, cached_idx, address_table, num_entries):
    sample.fill_neighbor_cache(cache, rowptr, col, cached_idx, address_table, num_entries)
