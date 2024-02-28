import csv
import pandas as pd
import torch
import numpy as np
a = torch.tensor([1,2,3])
b = torch.tensor([2, 3])
c = torch.tensor([[1, 3], [4, 5], [2, 7]])
cache_no_hit = a == 1
print (a[cache_no_hit].shape[0])
print (len(cache_no_hit))
a = np.array(a.tolist())
for b_single in b:
    print (b_single)
indice = [0, 1]
c[indice] = torch.tensor([[2, 3], [5, 2]])
print (c)
print (a[~cache_no_hit])