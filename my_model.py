import dival
import torch
import torch.nn as nn
import numpy as np
import random
from dival.measure import MSEMeasure,PSNRMeasure
from op import *

IMPL = 'astra_cuda'
#deep operator
def DO(net1,net2, dataloader,optimizer1,optimizer2,
            device,test=False,apply=False,
            batch_size = 16,
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
        gt = torch.unsqueeze(gt,1).to(device)
        y = torch.unsqueeze(y,1).to(device)
        y0 = op_adjoint(y,batch_size=batch_size)
        y=y0
        s = step_size

        iter = random.randint(0,n_iter)

        #training process
        for k in range(n_iter):
            n = net1.forward(y)
            y = y+ s*y0- s*n -s*(net2.forward(y))

            if k == iter:
                n = net1.forward(y)

                with torch.no_grad():
                    term = op(y,batch_size)
                    term = op_adjoint(term,batch_size)
                loss_1 = loss_fn(term,n).to(device)#deep operator loss

        loss_2 = loss_fn(y,gt).to(device)#supervised loss

        loss = loss_1 + loss_2
        loss = loss.to(device)

        with torch.no_grad():
            short_name = f'psnr{i}'
            gt= gt.cpu()
            y = y.cpu()
            PSNR = PSNRMeasure(short_name=short_name).apply(y,gt)
            y = y.to(device)
            gt = gt.to(device)
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

def DU(net, dataloader,optimizer,
            device,batch_size,test=False,
            n_iter=10,step_size=0.4):
    loss_seq = []
    loss_fn = nn.MSELoss()
    i=0
    for y, gt in dataloader:
        i += 1
        loss = 0
        gt = torch.unsqueeze(gt, 1).to(device)
        y = torch.unsqueeze(y, 1).to(device)
        y0 = op_adjoint(y, batch_size=batch_size)
        y=y0
        s = step_size
        iter = random.randint(0, n_iter)

        # training process
        for k in range(n_iter):
            with torch.no_grad():
                n = op(y,batch_size=batch_size)
                n = op_adjoint(n,batch_size)
            y = y + s * y0 - s * n - s * (net.forward(y))

        loss = loss_fn(y, gt).to(device)  # supervised loss

        with torch.no_grad():
            short_name = f'psnr{i}'
            gt = gt.cpu()
            y = y.cpu()
            PSNR = PSNRMeasure(short_name=short_name).apply(y, gt)
            y = y.to(device)
            gt = gt.to(device)
        print(f'{i}', 'Total Loss', loss, "PSNR=", PSNR)

        loss_seq.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss = np.mean(loss_seq)

    return loss