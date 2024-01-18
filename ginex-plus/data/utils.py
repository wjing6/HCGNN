from wrapper import *
import torch

def load_edge_list(edge_list_file_path, skip):
    return loader.load_edge_list(edge_list_file_path, skip)

def load_edge_list_from_binary(edge_list_binary_path, skip):
    return loader.load_edge_list_from_binary(edge_list_file_path, skip)

