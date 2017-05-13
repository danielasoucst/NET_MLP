import numpy as np
class Neuronio:
    def __init__(self, id,lstX,y,lstW,valor_teta):
        self.id = id
        self.y = y
        self.lstW = lstW
        self.valor_teta = valor_teta
        self.lstX = lstX


    def ativar(self):
        x = 0
        for i in range(0,len(self.lstX)):
            x += self.lstX[i]*self.lstW[i]
        x += (-1)*self.valor_teta
        sigmoid = 1/(1+np.exp((-1)*x))
        self.y = sigmoid
        return sigmoid


