import torch
import random
import numpy as np
p = torch.tensor([0], dtype=torch.int16)
msb = (torch.tensor([1], dtype=torch.int16) << 15)
print (p)
p |= msb
print (p)
