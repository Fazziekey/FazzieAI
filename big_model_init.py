import torch
from torch import nn
import time

import psutil
import os

import torch

def count_parameters(model):
    total_params =  sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params / 1e9 


def model_memory_size_in_megabytes(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()  

    bytes_in_gb = 1024 * 1024 * 1024 
    return param_size / bytes_in_gb



# 打印当前进程的CPU和内存使用情况
def print_usage():
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print(f'memory: {memory_use:.2f} GB')
    print('CPU percent:', psutil.cpu_percent())


class BigModel(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(size, size) for i in range(10)])

    def forward(self, x):
        for linear in self.linears:
            x = linear(x)
        return x

size = 100000
model = BigModel(size)

# # 打印模型的参数量
print(f'The model has {count_parameters(model):,} B trainable parameters')
print(f"The model's memory size is approximately {model_memory_size_in_megabytes(model):.2f} GB.")
torch.save(model.state_dict(), 'checkpoint.pth')

del model

print('Before Load the state_dict:')
print_usage()

start_time = time.time()
state_dict = torch.load('checkpoint.pth')
print(f'Loading the state_dict took {time.time() - start_time:.2f} seconds')
print('After Load the state_dict:')
print_usage()

start_time = time.time()
model = BigModel(size)
print(f'Init the model took {time.time() - start_time:.2f} seconds')
print('After Init the model:')
print_usage()

start_time = time.time()
model.load_state_dict(state_dict)
print(f'Loading the state_dict to model took {time.time() - start_time:.2f} seconds')
print('After Load the state_dict to model:')
print_usage()


print('Before Load the state_dict:')
print_usage()

start_time = time.time()
state_dict = torch.load('checkpoint.pth', mmap=True)
print(f'Loading the state_dict took {time.time() - start_time:.2f} seconds')
print('After Load the state_dict:')
print_usage()

start_time = time.time()
with torch.device('meta'):
  model = BigModel(size)
print(f'Init the model took {time.time() - start_time:.2f} seconds')
print('After Init the model:')
print_usage()

start_time = time.time()
model.load_state_dict(state_dict, assign=True)
print(f'Loading the state_dict to model took {time.time() - start_time:.2f} seconds')
print('After Load the state_dict to model:')
print_usage()


input = torch.randn(1, size)

output = model(input)