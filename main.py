# coding: utf-8
import neuronio
import random as rand
import copy
Fi = 2 #úmero total de entradas do neurônio i na rede.
taxa_aprendizagem = 0.1

def gerarPesosAleatorios():
    lstPesos = []

    for i in range(0,Fi):
        lstPesos.append(rand.uniform(-2.4/Fi,2.4/Fi))

    return lstPesos



#Inicializando

entrada = [1,1]
saida_esperada = [0]

'''n3 = neuronio.Neuronio(3,[],0,gerarPesosAleatorios(),rand.uniform(-2.4/Fi,2.4/Fi))
n4 = neuronio.Neuronio(4,[],0,gerarPesosAleatorios(),rand.uniform(-2.4/Fi,2.4/Fi))
n5 = neuronio.Neuronio(5,[],0,gerarPesosAleatorios(),rand.uniform(-2.4/Fi,2.4/Fi))'''

n3 = neuronio.Neuronio(3,[],0,[0.5,0.4],0.8)
n4 = neuronio.Neuronio(4,[],0,[0.9,1],-0.1)
n5 = neuronio.Neuronio(5,[],0,[-1.2,1.1],0.3)

camadas = [[n3,n4],[n5]]

for i in range(0,len(camadas)):
    qtdeNeuCam = len(camadas[i])
    if(qtdeNeuCam>1):
        for j in range(0,len(camadas[i])):
            neuronio = camadas[i][j]
            neuronio.lstX = copy.copy(entrada)
            print (neuronio.id,neuronio.ativar())
        #atualizar entrada
        for k in range(0,len(entrada)):
            entrada[k] = camadas[i][k].y
    else:#Quando temos um neuronio na camada
        neuronio = camadas[i][0]
        neuronio.lstX = copy.copy(entrada)
        print (neuronio.ativar(),entrada)
        for k in range(0,len(entrada)):
            entrada[i] = neuronio.y
erro = saida_esperada - camadas[len(camadas)-1][0].y
print("erro: ",erro[0])

print(camadas[0][0].atualizar_pesos(erro,taxa_aprendizagem))
print(camadas[0][1].atualizar_pesos(erro,taxa_aprendizagem))
print(camadas[1][0].atualizar_pesos(erro,taxa_aprendizagem))









