import torch
import random
import numpy as np
edge_index = torch.Tensor([[1, 0 ,2, 4], [0, 0, 2, 5]])
mask = [True, True, False, False]
edge_index = edge_index[:, mask]
print (edge_index)
