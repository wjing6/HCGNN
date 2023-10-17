import struct
import torch
import os
from tqdm import tqdm
import time
class edgeList():
    def __init__(self, filename = None):
        if filename:
            self.filename = filename
            self.maxId = 0

    def getEdgeIndex(self):
        edge_list = []  # uint32_t
        read_start = time.time()
        f = open(self.filename)
        line = f.readline()
        step = 0
        while (line):
            line = line.split()
            src = int(line[0])
            dst = int(line[1])
            line = f.readline()
            if src > self.maxId:
                self.maxId = src
            if dst > self.maxId:
                self.maxId = dst
            edge_list.append([src, dst])
        
        edge_index = torch.tensor(edge_list).t()
        time_floating = time.time() - read_start
        print ("cost {:.4f} s".format(time_floating))
        return edge_index, self.maxId
