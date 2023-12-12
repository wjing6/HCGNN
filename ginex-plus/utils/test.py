import torch

dataset = range(1024)
dataloader = torch.utils.data.DataLoader(dataset,
                                        batch_size=2,
                                        shuffle=True)

for iter, seeds in enumerate(dataloader):
    continue

sample_loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=512,
                                        shuffle=True)

for iter, seeds in enumerate(sample_loader):
    print (seeds)