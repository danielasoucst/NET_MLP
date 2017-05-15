import numpy as np

class Neuronio:
    def __init__(self, id,lstX,y,lstW,valor_teta):
        self.id = id
        self.y = y
        self.lstW = lstW
        self.valor_teta = valor_teta
        self.lstX = lstX


    def ativar(self,opt):
        x = 0
        for i in range(0,len(self.lstX)):
            x += self.lstX[i]*self.lstW[i]
        x -= self.valor_teta

        if(opt == 1):
            self.y = self.linear_activation(x)
        if(opt == 2):
            self.y = self.sigmoid_activation(x)
        if(opt == 3):
            self.y = self.hyperbolic_tangent_activation(x)
        if(opt == 4):
            self.y = self.gaussian_activation(x)
        return self.y


    def linear_activation(self,x):
        return x

    def sigmoid_activation(self,x):
        sigmoid = 1 / (1 + np.exp((-1) * x))
        return sigmoid

    def hyperbolic_tangent_activation(self,x):

        ht = (2/(1+np.exp((-2) * x)))-1
        return ht

    def gaussian_activation(self,x):
        gs = np.exp((-1)*(x**2))
        return gs

