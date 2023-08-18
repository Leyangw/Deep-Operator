from Unet import *
from my_model import *

import os
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Adam


# lr = 0.01
#n_iter = 10
#epochs = 1000

class train_MyModel(object):
    def __init__(self,
                 in_chans, out_chans,
                 chans_net1,chans_net2,
                 num_pool_layers_net1,num_pool_layers_net2,
                 drop_prob,device):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans_net1 = chans_net1
        self.chans_net2 = chans_net2
        self.num_pool_layers_net1 = num_pool_layers_net1
        self.num_pool_layers_net2 = num_pool_layers_net2
        self.drop_prob = drop_prob
        self.device = device

    def train_DO(self,Dataloader,epochs,ckp_interval,
              step_size,
              lr_rate1,lr_rate2,
              batch_size,
              save_path, test_data,test = False):

        generator_1 = UnetModel(self.in_chans,self.out_chans,self.chans_net1,self.num_pool_layers_net1,self.drop_prob).to(self.device)

        optimizer1 = Adam(list(generator_1.parameters()),lr= lr_rate1,weight_decay=1e-8)

        generator_2 = UnetModel(self.in_chans, self.out_chans, self.chans_net2, self.num_pool_layers_net2,
                                    self.drop_prob).to(self.device)
        optimizer2 = Adam(list(generator_2.parameters()), lr=lr_rate2, weight_decay=1e-8)

        for epoch in range(epochs):
            print(f'-----{epoch}-----------')
            loss = DO(net1 = generator_1,net2 = generator_2,
                dataloader = Dataloader,
                optimizer1=optimizer1,optimizer2=optimizer2,
                device= self.device,batch_size=batch_size,
                n_iter=10,step_size = step_size)

            if epoch % ckp_interval == 0 or epoch + 1 == epochs:
                state = {'epoch': epoch,
                             'state_dict_1': generator_1.state_dict(),
                             'state_dict_2': generator_2.state_dict(),
                             'optimizer_1': optimizer1.state_dict(),}
                torch.save(state, os.path.join(save_path, 'ckp_{}.pth.tar'.format(epoch)))
                if test:
                    with torch.no_grad():
                        loss = DO(net1 = generator_1,net2 = generator_2,
                        dataloader = Dataloader,
                        optimizer1=optimizer1,optimizer2=optimizer2,
                        device= self.device,batch_size=batch_size,
                        n_iter=10,step_size = step_size)

    def train_DU(self,Dataloader,epochs,ckp_interval,
              step_size,lr_rate,batch_size,
              save_path, test_data,test = False):
        generator_1 = UnetModel(self.in_chans, self.out_chans, self.chans_net1, self.num_pool_layers_net1,
                                self.drop_prob).to(self.device)

        optimizer1 = Adam(list(generator_1.parameters()), lr=lr_rate, weight_decay=1e-8)
        for epoch in range(epochs):
            print(f'-----{epoch}-----------')
            loss = DU(net=generator_1,
                    dataloader=Dataloader,
                    optimizer=optimizer1,
                    device=self.device, batch_size=batch_size,
                    n_iter=10, step_size=step_size)

            if epoch % ckp_interval == 0 or epoch + 1 == epochs:
                state = {'epoch': epoch,
                        'state_dict_1': generator_1.state_dict(),
                        'optimizer_1': optimizer1.state_dict(), }
                torch.save(state, os.path.join(save_path, 'ckp_{}.pth.tar'.format(epoch)))
                if test:
                    with torch.no_grad():
                         loss = DU(net=generator_1,
                                  dataloader=test_data,
                                  optimizer=optimizer1,
                                  device=self.device, batch_size=batch_size,
                                  n_iter=10, step_size=step_size)