import numpy as np
class Neuronio:
    def __init__(self, id,lstX,y,lstW,valor_teta,nextNeuronio,isSaida):
        self.id = id
        self.y = y
        self.lstW = lstW
        self.valor_teta = valor_teta
        self.lstX = lstX
        self.nextNeuronio = nextNeuronio
        self.isSaida = isSaida

    def ativar(self):
        x = 0
        for i in range(0,len(self.lstX)):
            x += self.lstX[i]*self.lstW[i]
        x += (-1)*self.valor_teta
        sigmoid = 1/(1+np.exp((-1)*x))
        self.y = sigmoid
        return sigmoid

    def atualizar_pesos(self,erro,taxa_aprendizagem):
        if(self.isSaida):
            gradiente = self.y * (1 - self.y) * erro
        else:
            gradiente = self.y * (1 - self.y) * self.nextNeuronio.
        print("gradiente: ",self.lstX,gradiente)
        self.valor_teta += taxa_aprendizagem * (-1) * gradiente
        print("teta: ",self.valor_teta)
        for i in range(0,len(self.lstX)):
            self.lstW[i] += taxa_aprendizagem * self.lstX[i] * gradiente

        print self.lstW
