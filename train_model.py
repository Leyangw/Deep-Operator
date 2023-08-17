from Unet import *
from my_model import *

import torch
import torch.nn as nn
from torch.optim import Adam


# lr = 0.01
#n_iter = 10
#epochs = 1000

class train_MyModel(object):
    def __init__(self,
                 in_chans, out_chans,chans, num_pool_layers, drop_prob,
                 device):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.device = device

    def train(self,Dataloader,epochs,ckp_interval,
              step_size,model,
              lr_rate1,lr_rate2,
              train_data,batch_size,
              save_path, test_data):

        generator_1 = UnetModel(self.in_chans,self.out_chans,self.chans,self.num_pool_layers,self.drop_prob).to(self.device)

        generator_2 = UnetModel(self.in_chans, self.out_chans,self.chans,self.num_pool_layers,self.drop_prob).to(self.device)

        optimizer1 = Adam(list(generator_1.parameters()),lr= lr_rate1,weight_decay=1e-8)
        optimizer2 = Adam(list(generator_2.parameters()),lr= lr_rate2,weight_decay=1e-8)
        if model == 'DO':
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
        elif model == 'DU':
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