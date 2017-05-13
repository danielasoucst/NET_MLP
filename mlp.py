import neuronio
import random as rand
import numpy as np

Fi = 2
class Mlp:
    def __init__(self,inputVect,output,taxa_aprendizagem):
        self.lstX = inputVect #[1,0] n0,n1
        self.desiredOutput = output
        self.error = 0

        n2 = neuronio.Neuronio(2, inputVect, 0, [0.5, 0.4], 0.8)
        n3 = neuronio.Neuronio(3, inputVect, 0, [0.9, 1], -0.1)
        n4 = neuronio.Neuronio(4,[0,0],0,[-1.2,1.1],0.3)

        #n2 = neuronio.Neuronio(2, inputVect, 0, self.createRandomWeights(), rand.uniform(-2.4 / Fi, 2.4 / Fi))
        #n3 = neuronio.Neuronio(3, inputVect, 0, self.createRandomWeights(), rand.uniform(-2.4 / Fi, 2.4 / Fi))
        #n4 = neuronio.Neuronio(4, [], 0, gerarPesosAleatorios(), rand.uniform(-2.4 / Fi, 2.4 / Fi))

        self.hiddenLayer = [n2,n3]
        self.outputLayer = n4
        self.alfa = taxa_aprendizagem
        self.matWeight = np.zeros((5,5), dtype=np.float)
        self.update_weight_matrix()

    def createRandomWeights(self):
        lstPesos = []

        for i in range(0, Fi):
            lstPesos.append(rand.uniform(-2.4 / Fi, 2.4 / Fi))

        return lstPesos

    def update_weight_matrix(self):
        self.matWeight[0][2] = self.hiddenLayer[0].lstW[0]
        self.matWeight[1][2] = self.hiddenLayer[0].lstW[1]
        self.matWeight[0][3] = self.hiddenLayer[1].lstW[0]
        self.matWeight[1][3] = self.hiddenLayer[1].lstW[1]

        self.matWeight[2][4] = self.outputLayer.lstW[0]
        self.matWeight[3][4] = self.outputLayer.lstW[1]

    def activate_neurons(self):
        for i in range(0,len(self.hiddenLayer)):
            new_y = self.hiddenLayer[i].ativar()
            self.outputLayer.lstX[i] = new_y

        self.outputLayer.ativar()

    def calculate_weight_matrix(self):
        grad_4 = self.outputLayer.y*(1-self.outputLayer.y)*self.error
        grad_2 = self.hiddenLayer[0].y*(1-self.hiddenLayer[0].y)*grad_4*self.matWeight[2][4]
        grad_3 = self.hiddenLayer[1].y*(1-self.hiddenLayer[1].y)*grad_4*self.matWeight[3][4]

        delta_weight4 = [self.alfa*self.hiddenLayer[0].y*grad_4,self.alfa*self.hiddenLayer[1].y*grad_4]
        delta_weight2 = [self.alfa * self.hiddenLayer[0].lstX[0] * grad_2, self.alfa * self.hiddenLayer[0].lstX[1] * grad_2]
        delta_weight3 = [self.alfa * self.hiddenLayer[1].lstX[0] * grad_3, self.alfa * self.hiddenLayer[1].lstX[1] * grad_3]

        delta_teta4 = self.alfa*(-1)*grad_4
        delta_teta2 = self.alfa*(-1)*grad_2
        delta_teta3 = self.alfa*(-1)*grad_3

        self.hiddenLayer[0].lstW = [sum(x) for x in zip(self.hiddenLayer[0].lstW, delta_weight2)]
        self.hiddenLayer[1].lstW = [sum(x) for x in zip(self.hiddenLayer[1].lstW, delta_weight3)]
        self.outputLayer.lstW = [sum(x) for x in zip(self.outputLayer.lstW, delta_weight4)]

        self.hiddenLayer[0].valor_teta += delta_teta2
        self.hiddenLayer[1].valor_teta += delta_teta3
        self.outputLayer.valor_teta += delta_teta4


    def train(self):
        ep = 0
        while(1):
            self.activate_neurons()
            self.error = self.desiredOutput - self.outputLayer.y
            if((self.desiredOutput-self.error)**2 < 0.001):
                break
            else:
                self.calculate_weight_matrix()
                self.update_weight_matrix()
            ep += 1
        return ep

    def test(self,inputTest):
        self.lstX = inputTest
        self.activate_neurons()
        return self.outputLayer.y


    def train2(self):
        ep = 0

        self.activate_neurons()
        self.error = self.desiredOutput - self.outputLayer.y
        self.calculate_weight_matrix()
        self.update_weight_matrix()














