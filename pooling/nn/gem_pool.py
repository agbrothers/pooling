import torch
import torch.nn as nn
from torch import Tensor, BoolTensor, log, sum, exp, mean

from pooling.nn.pool import Pool



class GeneralizedMean(Pool): 
    """
    DESC: 
    An implementation of the Generalized Mean as described 
    in the paper `Generalized Mean Attention`. 
    
    """

    def __init__(self, p=1, **kwargs):
        super().__init__(dim=-2)
        self.p = p

    def M_p(self, X):
        return mean( X**self.p, axis=-2) ** (1/self.p)

    def forward(self, X:Tensor, mask:BoolTensor=None): 
        return self.M_p(X)
    

class KolmogorovMean(Pool): 
    """
    DESC: 
    An implementation of the Generalized f-Mean as described 
    in the paper `Generalized Mean Attention`. 
    
    """

    def __init__(self, p=1, b=0.9, **kwargs):
        super().__init__(dim=-2)
        self.p = p
        self.b = b
        self.eps = 1e-10
        self.p_min = 1e-6
        self.p_scale = 1e+4
        self.X_min = None
        self.X_max = None
        self.Z_max = None

    def norm(self, x):
        return (1 - self.b) * ( x-self.X_min) / (self.X_max - self.X_min + self.eps) + self.b
    
    def norm_inv(self, y):
        return (y - self.b) * (self.X_max - self.X_min) / (1 - self.b) + self.X_min

    def f(self, x):
        return exp( self.p * log( self.norm(x) ) - self.Z_max )
    
    def f_inv(self, y):
        return self.norm_inv( exp( 1/self.p * (self.Z_max + log(y)) ) )
    
    def M_f(self, X):
        return self.f_inv( mean( self.f(X), axis=-2) )

    def forward(self, X:Tensor, mask:BoolTensor=None): 
        self.X_max = torch.max(X, axis=-2).values
        self.X_min = torch.min(X, axis=-2).values
        self.Z_max = torch.max(self.p * log(self.norm(X)), axis=-2).values
        return self.M_f(X)
    
    # def get_p(self):
    #     return self.circular_clamp(nn.tanh(self.p * self.p_scale))

    # def circular_clamp(self, x):
    #     ## GET SIGN. IF x=0, FORCE SIGN TO BE POSITIVE
    #     sign = (x.sign()+self.p_min).sign() 
    #     return sign * torch.maximum(x.abs(), torch.Tensor([self.p_min])) 



if __name__ == "__main__":

    ## TEST DATA
    d = 8
    a = torch.tensor([[0, 1, 2, 3, 4, 5]]).T
    X = torch.tile(a, (1, d))
    mask = torch.BoolTensor([0,1,0,0,1,0])

    p = torch.arange(0, d, 1, dtype=torch.float32)
    p[0] = -1e+5
    p[-1] = 1e+5
    gem = GeneralizedMean(p=p)
    kom = KolmogorovMean(p=p, b=1e-6)

    cubic_g = gem(X)
    cubic_k = kom(X)

    catch = True


    ## CHECK THAT MEANS ARE NOT BIAS BY PROXIMITY TO ZERO FOR ANY b
    ## b = 0.01
    pos = torch.Tensor([2.5000, 3.0190, 3.3380, 3.5651, 3.7373, 3.8732, 3.9836, 4.0751])
    neg = torch.Tensor([-2.5000, -1.9810, -1.6620, -1.4349, -1.2627, -1.1268, -1.0164, -0.9249])
    for i in range(len(pos)):
        assert abs(pos[0] - pos[i]) - abs(neg[0] - neg[i]) < 1e-8

    pos = torch.Tensor([2.5000, 3.0276, 3.3471, 3.5739, 3.7458, 3.8813, 3.9913, 4.0824])
    neg = torch.Tensor([-2.5000, -1.9724, -1.6529, -1.4261, -1.2542, -1.1187, -1.0087, -0.9176])
    for i in range(len(pos)):
        print(abs(pos[0] - pos[i]) - abs(neg[0] - neg[i]), abs(pos[0] - pos[i]) , abs(neg[0] - neg[i]))
        assert abs(abs(pos[0] - pos[i]) - abs(neg[0] - neg[i])) < 1e-6
