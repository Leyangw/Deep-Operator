import torch
import torch.nn as nn

import random
import odl.contrib.torch.operator as odl_torch

class op(nn.Module):

    def __init__(self,forward_op):
        super(op,self).__init__()
        self.forward_op = forward_op
        self.op_layer = odl_torch.OperatorModule(self.forward_op)

    def forward(self,input):
        return self.op_layer(input)

class op_adjoint(nn.Module):

    def __init__(self,forward_op):
        super(op_adjoint,self).__init__()
        self.forward_op = forward_op
        self.opadjoint_layer = odl_torch.OperatorModule(self.forward_op.adjoint)

    def forward(self,input):
        return self.opadjoint_layer(input)


class DO_Grad(nn.Module):

    def __init__(self,net1,net2,forward_op,n_iter=10,step_size=0.4):
        super(DO_Grad, self).__init__()
        self.n_iter = n_iter
        self.s = step_size
        self.net1 = net1
        self.net2 = net2
        self.forward_op = forward_op
        self.op_layer = odl_torch.OperatorModule(self.forward_op)
        self.opadjoint_layer = odl_torch.OperatorModule(self.forward_op.adjoint)

    def approx(self,input):
        approximate = self.net1(input)
        return approximate

    def forward(self,input):
        y = self.opadjoint_layer(input)
        y0 = y
        for iter in range(self.n_iter):
            y = y + self.s * y0 - self.s * (self.approx(y)) - self.s * (self.net2(y))

        return y

    def term(self):
        return self.term


class DO_PROX(nn.Module):

    def __init__(self,forward_op):
        self.forward_op = forward_op

class DU_Grad(nn.Module):

    def __init__(self,net,forward_op,n_iter=10,step_size=0.4):
        super(DU_Grad, self).__init__()
        self.n_iter = n_iter
        self.s = step_size
        self.net = net
        self.forward_op = forward_op
        self.op_layer = odl_torch.OperatorModule(self.forward_op)
        self.opadjoint_layer = odl_torch.OperatorModule(self.forward_op.adjoint)

    def forward(self,input):
        y = self.opadjoint_layer.forward(input)
        y0 = y
        for i in range(self.n_iter):
            y = y + self.s * y0 - self.s * (self.opadjoint_layer(self.op_layer(y))) - self.s * (self.net(y))
        return y

class approx_fn(nn.Module):

    def __init__(self,forward_op):
        super(approx_fn,self).__init__()
        self.forward_op = forward_op
        self.op_layer = odl_torch.OperatorModule(self.forward_op)
        self.opadjoint_layer = odl_torch.OperatorModule(self.forward_op.adjoint)
        self.layer = nn.Sequential(
            self.op_layer,
            self.opadjoint_layer
        )

    def forward(self,input):
        output = self.layer(input)
        return output
