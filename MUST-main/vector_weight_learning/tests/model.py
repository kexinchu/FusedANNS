import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.optim import Optimizer

k = 1
t = k * 1

class AggregationFunctionModel(nn.Module):

    def __init__(self):
        super(AggregationFunctionModel, self).__init__()

        self.omega_x1 = nn.Parameter(Variable(torch.FloatTensor([0.5]), requires_grad = True))
        self.omega_x2 = nn.Parameter(Variable(torch.FloatTensor([0.5]), requires_grad = True))
        self.omega_x3 = nn.Parameter(Variable(torch.FloatTensor([0.5]), requires_grad = True))
        self.omega_x4 = nn.Parameter(Variable(torch.FloatTensor([0.5]), requires_grad = True))

    def forward(self, d_x, d_y, d_z, d_v):
        return self.omega_x1 * d_x + self.omega_x2 * d_y + self.omega_x3 * d_z + self.omega_x4 * d_v


class AggregationFunctionLoss(nn.Module):

    def __init__(self) -> None:
        super(AggregationFunctionLoss, self).__init__()

    def forward(self, phi_positive, phi_negative):
        # Use a numerically stable form: log(1 + exp((neg - pos) / t))
        # This avoids inf/NaN when distances are large.
        loss_terms = []
        for pos, neg in zip(phi_positive, phi_negative):
            loss_terms.append(torch.sum(torch.nn.functional.softplus((neg - pos) / t)))
        loss = (1 / len(loss_terms)) * sum(loss_terms)
        print("loss: ", loss)
        return loss

class AggregationFunctionOptimizer(Optimizer):

    def __init__(self, params, lr):
        self.lr = lr
        super(AggregationFunctionOptimizer, self).__init__(params, {})

    def step(self, closure=False):
        random_num = 0.4

        for param_group in self.param_groups:
            params = param_group['params']
            for param in params:
                param.data = param.data - self.lr * random_num * param.grad
