import torch

print(torch.cuda.is_available())

print(torch.cuda.get_device_properties(0).total_memory)

print(torch.cuda.mem_get_info(0))