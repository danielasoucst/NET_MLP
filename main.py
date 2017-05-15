# coding: utf-8
import mlp
import numpy as np
''' Função de Ativação
1 - Linear
2 - Sigmoid
3 - Hiperbolica tangente
4 - Gaussiana
'''
OPT_FUNCAO =4
TAXA_APRENDIZAGEM = 0.1
MAX_EPOCAS = 30000

#matTrain = [[1,1,0],[0,0,0],[0,1,1],[1,0,1]]
'''fileTrain = open('train.txt','r')
treino = fileTrain.readlines()

lstAcertos = []
lstEpocas = []
matTest = [[0,1,1],[1,0,1],[1,1,0],[0,0,0]]

def getEstatisticas(vetEpocas,vetAcertos):
    cont = 0
    for e in vetEpocas:
        if(e>=MAX_EPOCAS):
            cont += 1

    epocasMedia = np.sum(vetEpocas)/24
    acertosMedia = np.sum(vetAcertos)*100/(24*4)
    return [epocasMedia,acertosMedia,cont]

for line in treino:
    vec = line.split(' ')
    aux = 0
    matTrain= []
    for i in range(0,len(vec)/3):
        t = [int(vec[aux]),int(vec[aux+1]),int(vec[aux+2])]
        matTrain.append(t)
        aux += 3
    #print(matTrain)
    m = mlp.Mlp(TAXA_APRENDIZAGEM, OPT_FUNCAO,MAX_EPOCAS)

    epocas = m.train(matTrain)
    #print(matTrain,epocas)
    lstEpocas.append(epocas)

    acertos = 0
    for i in range(0,len(matTest)):
        saida = m.test(matTest[i][:2])
        if(saida==matTest[i][2]):
            acertos += 1
    lstAcertos.append(acertos)

print(lstEpocas)
print(lstAcertos)
resu = getEstatisticas(lstEpocas,lstAcertos)
print("Epocas", resu[0],"Acertos",resu[1],"Casos nao conver.",resu[2])'''



matTrain = [[0,1,1],[0,0,0],[1,0,1],[1,1,0]]
#matTrain = [[1,1,0],[0,1,1],[0,0,0],[1,0,1]]
m = mlp.Mlp(TAXA_APRENDIZAGEM, 4,MAX_EPOCAS)
epo = m.train(matTrain)

print("Terminou treinamento",epo)

print("test",m.test([0,1]))
print("test",m.test([1,1]))
print("test",m.test([1,0]))
print("test",m.test([0,0]))


#print vect[0]




''''''


'''Inicialização da rede mlp'''
#m = mlp.Mlp(TAXA_APRENDIZAGEM,OPT_FUNCAO)

'''Treinamento da rede'''
#epocas = m.train(matTrain)

#print("Terminou treinamento",epocas)

'''Teste'''
'''acertos = 0
for i in range(0,len(matTest)):
    saida = m.test(matTest[i][:2])
    if(saida==matTest[i][2]):
        acertos+=1

print("Acertos:",acertos)'''

