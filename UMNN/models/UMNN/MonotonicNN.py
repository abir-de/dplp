import torch
import torch.nn as nn
from .NeuralIntegral import NeuralIntegral
from .ParallelNeuralIntegral import ParallelNeuralIntegral
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


def _flatten(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


class custom_linear(Module):
    
    def __init__(self, in_features, out_features):
    
        super(custom_linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
        
        
    def reset_parameters(self):
        
       # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        #nn.init.uniform_(self.weight, a=-300.0, b=300.0)
        self.weight.data =torch.ones(self.in_features, self.out_features) #(torch.abs(self.weight.data))**2
        nn.init.constant_(self.bias, 0)
 
 
    def forward(self, x):
        w = self.weight
        output = torch.mm(x,w)
        return  output+self.bias



class IntegrandNN(Module):
    def __init__(self, in_d, hidden_layers):
        super(IntegrandNN, self).__init__()
        self.net = []
        hs = [in_d] + hidden_layers + [1]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                custom_linear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        self.net.append(nn.ELU())
        self.net = nn.Sequential(*self.net)

    def forward(self, x, h):
        return self.net(torch.cat((x, h), 1)) + 1.

class MonotonicNN(Module):
    def __init__(self, in_d, hidden_layers, nb_steps=50, dev="cpu"):
        super(MonotonicNN, self).__init__()
        self.integrand = IntegrandNN(in_d, hidden_layers)
        self.net = []
        hs = [in_d-1] + hidden_layers + [2]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                custom_linear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        # It will output the scaling and offset factors.
        self.net = nn.Sequential(*self.net)
        self.device = dev
        self.nb_steps = nb_steps

    '''
    The forward procedure takes as input x which is the variable for which the integration must be made, h is just other conditionning variables.
    '''
    def forward(self, x, h):
        x0 = torch.zeros(x.shape).to(self.device)
        out = self.net(h)
        offset = out[:, [0]]
        scaling = torch.exp(out[:, [1]])
        return scaling*ParallelNeuralIntegral.apply(x0, x, self.integrand, _flatten(self.integrand.parameters()), h, self.nb_steps) + offset
