import dival
import torch
import torch.nn as nn
import numpy as np
import random
from dival.measure import MSEMeasure,PSNRMeasure
from op import *

IMPL = 'astra_cuda'
dataset = dival.get_standard_dataset('LoDopab',impl=IMPL)
#deep operator
def DO(net1,net2, dataloader,optimizer1,optimizer2,
            device,test=False,batch_size = 16,
            n_iter=10,step_size=0.4):
    if test:
        test_error = []
    loss_seq,loss_measuer_seq= [],[]
    loss_fn = nn.MSELoss()
    i = 0
    for y,gt in dataloader:
        i+=1
        loss_1 = 0
        loss_2 = 0
        gt = torch.unsqueeze(gt,1)
        y = torch.unsqueeze(y,1).to(device)
        y = op_adjoint(y,batch_size).to(device)
        s = step_size
        iter = random.randint(0,n_iter)

        for k in range(n_iter):
            n = net1.forward(y)
            y = y+ s*y- s*n -s*(net2.forward(y))
            if k == iter:
                n = net1.forward(y)
                with torch.no_grad():
                    term = op(y,batch_size)
                    term = op_adjoint(term,batch_size)
                loss_1 = loss_fn(term,n)#deep operator loss
        loss_2 = loss_fn(y,gt)#supervised loss
        loss = loss_1 + loss_2
        with torch.no_grad():
            short_name = f'psnr{i}'
            PSNR = PSNRMeasure(short_name=short_name).apply(y,gt)
        print(f'{i}','L1=',loss_1,'L2=',loss_2,'Total Loss',loss,"PSNR=",PSNR)
        if loss_1 != 0:
            loss_seq.append(loss_1.item())
        loss_measuer_seq.append(loss_2.item())

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        loss.backward()

        optimizer1.step()
        optimizer2.step()

    loss = [np.mean(loss_seq),np.mean(loss_measuer_seq)]

    return loss

def DU(net, dataloader, physics,
       optimizer,
        criterion_loss,
        dtype, device,
        n_iter,step_size):
    loss_seq = []

    for i, x in enumerate(dataloader):
        x = x[0] if isinstance(x, list) else x

        x = x.type(dtype).to(device)  # ground-truth signal x

        y = physics.A(x.type(dtype).to(device))  # generate measurement input y
        x0 = physics.A_dagger(y)
        x1 = x0
        s = step_size

        for i in range(n_iter):
            x1 = x1 + s * x0 - s*physics.A_dagger(physics.A(x1)) - s * net(x1)

        loss = criterion_loss(x,x1)

        loss_seq.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss = np.mean(loss_seq)

    return loss