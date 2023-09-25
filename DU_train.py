import dival

from PIL import Image
import os

from train_model import *
from my_model import *
from dival import get_standard_dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:21"

def main():
    batch_size = 2
    device = torch.device("cuda")
    dataset = get_standard_dataset('LoDopab',impl='astra_cuda')
    train_data = dataset.create_torch_dataset(part='train')
    data = DataLoader(train_data,batch_size=batch_size)
    validation_data = dataset.create_torch_dataset(part='validation')
    test_data = DataLoader(dataset.create_torch_dataset(part='test'))

    train_MyModel(1,1,16,32,4,8,0,device).train_DU(Dataloader=data,epochs=1,
                                    ckp_interval= 10,step_size=0.4,lr_rate=0.01,
                                    batch_size=batch_size,save_path='C:/Users/w1642/Desktop',test_data=test_data)

    print('--------training end-------------')



if __name__ == '__main__':
    main()