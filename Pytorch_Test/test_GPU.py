# Pytorch Test

import torch


print("\n")
print("GPU: ",torch.cuda.is_available())
print("\n")



x = torch.rand(5, 3)

print(x)
