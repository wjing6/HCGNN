import csv
import pandas as pd
import torch
import numpy as np
a = torch.tensor([1, 2, 5, 8])
data = torch.tensor([5, 2, 7, 6])
idx = torch.tensor([0, 2])
res = torch.tensor([2, 8])
feature = torch.rand([10, 5])
print (feature)
push_feature = torch.rand([2, 5])
print (push_feature)
feature[idx] = push_feature
print (feature)