import torch
torch.zeros([20000, 4000], dtype=torch.int32)
cuda0 = torch.device('cuda:1')
x = torch.ones([20000, 40000], dtype=torch.float64, device=cuda0)
print(x)