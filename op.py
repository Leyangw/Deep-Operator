from dival import get_standard_dataset
import torch
import numpy as np

dataset = get_standard_dataset('LoDopab')
ray_trafo = dataset.get_ray_trafo()

def op(Tensor,batch_size=16,H=1000,W=513):
    with torch.no_grad():
        Tensor = torch.squeeze(Tensor,1)
        zero = torch.zeros([batch_size,H,W])
        for layer in range(batch_size):
            zero[layer] = torch.tensor(ray_trafo(Tensor[layer]).asarray())
        zero = torch.unsqueeze(zero,1)
    return zero

def op_adjoint(Tensor,batch_size=16,H=362,W=362):
    with torch.no_grad():
        Tensor = torch.squeeze(Tensor,1)
        zero = torch.zeros([batch_size,H,W])
        for layer in range(batch_size):
            zero[layer] = torch.tensor(ray_trafo.adjoint(Tensor[layer]).asarray(),dtype = torch.float)
        zero = torch.unsqueeze(zero, 1)
    return zero



