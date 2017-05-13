import mlp

matTrain = [[1,1,0],[0,0,0],[0,1,1],[1,0,1]]
m = mlp.Mlp(matTrain[0],0.1)
m.train(None)

for i in range(1,len(matTrain)):
    m.train(matTrain[i])
print("Terminou treinamento")

print("test",m.test([1,1]))

