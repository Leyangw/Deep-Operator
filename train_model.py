from Unet import *
from my_model import *

import os
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Adam
from dival.measure import PSNRMeasure
from dival import get_standard_dataset

ray_trafo = get_standard_dataset('LoDopab',impl='astra_cuda').get_ray_trafo()


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

    def train_DO(self, Dataloader,epochs,ckp_interval,
              step_size,
              lr_rate1,lr_rate2,
              batch_size,
              save_path, test_data,
              s = 0.4,n_iter = 10,test = False):

        generator_1 = UnetModel(self.in_chans,self.out_chans,self.chans_net1,self.num_pool_layers_net1,self.drop_prob).to(self.device)

        optimizer1 = Adam(list(generator_1.parameters()),lr= lr_rate1,weight_decay=1e-8)

        generator_2 = UnetModel(self.in_chans, self.out_chans, self.chans_net2, self.num_pool_layers_net2,
                                    self.drop_prob).to(self.device)
        optimizer2 = Adam(list(generator_2.parameters()), lr=lr_rate2, weight_decay=1e-8)
        op_adj = op_adjoint(ray_trafo)
        approx_Fn = approx_fn(ray_trafo).to(self.device)
        MSE = nn.MSELoss()
        short_name = 0
        for epoch in range(epochs):
            print(f'-----{epoch}-----------')
            for ob,gt in Dataloader:
                k = random.randint(0,n_iter-1)
                ob = torch.unsqueeze(ob,1).to(self.device)
                gt = torch.unsqueeze(gt,1).to(self.device)
                y0 = op_adj(ob)
                y = y0
                for iter in range(n_iter):
                    approx_term = generator_1(y)
                    y = y + s * y0 - s * approx_term- s * (generator_2(y))
                    if iter == k:
                        rand_term = y
                        approx = approx_Fn(rand_term)
                        loss1 = MSE(approx_term,approx).to(self.device)

                loss2 = MSE(y,gt).to(self.device)

                with torch.no_grad():
                    short_name +=1
                    t = y.cpu()
                    g = gt.cpu()
                    PSNR = PSNRMeasure(short_name=short_name).apply(t,g)

                print(f'--loss1:{loss1.item()}---loss2:{loss2.item()}---PSNR:{PSNR}')
                optimizer1.zero_grad()

                optimizer2.zero_grad()
                loss1.backward(retain_graph = True)

                loss2.backward()

                optimizer1.step()
                optimizer2.step()

            if epoch % ckp_interval == 0 or epoch + 1 == epochs:
                state = {'epoch': epoch,
                            'state_dict_1': generator_1.state_dict(),
                            'state_dict_2': generator_2.state_dict(),
                            'optimizer_1': optimizer1.state_dict(),
                            'optimizer_2': optimizer2.state_dict()
                         }
                torch.save(state, os.path.join(save_path, 'ckp_{}.pth.tar'.format(epoch)))
                if test:
                    for ob,gt in test_data:
                        with torch.no_grad():
                            loss = DO_Grad(net1 = generator_1,net2 = generator_2,
                            n_iter=10,step_size = step_size)

    def train_DU(self,Dataloader,epochs,ckp_interval,
              step_size,lr_rate,batch_size,
              save_path, test_data,test = False):
        generator_1 = UnetModel(self.in_chans, self.out_chans, self.chans_net1, self.num_pool_layers_net1,
                                self.drop_prob).to(self.device)

        optimizer1 = Adam(list(generator_1.parameters()), lr=lr_rate, weight_decay=1e-8)

        DU_GRAD = DU_Grad(net=generator_1,forward_op = ray_trafo)
        MSE = nn.MSELoss()

        for epoch in range(epochs):
            print(f'-----{epoch}---start--------')
            for ob,gt in Dataloader:

                ob = torch.unsqueeze(ob,1).to(self.device)
                gt = torch.unsqueeze(gt,1).to(self.device)

                reco_term = DU_GRAD(ob)

                loss = MSE(reco_term,gt).to(self.device)
                print(f'-----------loss:{loss.item()}')

                optimizer1.zero_grad()
                loss.backward()
                optimizer1.step()

            if epoch % ckp_interval == 0 or epoch + 1 == epochs:
                state = {'epoch': epoch,
                        'state_dict_1': generator_1.state_dict(),
                        'optimizer_1': optimizer1.state_dict(), }
                torch.save(state, os.path.join(save_path, 'ckp_{}.pth.tar'.format(epoch)))

                if test:
                    for ob,gt in enumerate(test_data):
                        with torch.no_grad():
                            term = DU_GRAD(ob)
                            loss = nn.MSELoss(term,gt)
                            PSNR = PSNRMeasure.apply(term,gt)