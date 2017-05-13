import mlp

matTrain = [[1,1,0],[0,0,0],[0,1,1],[1,0,1]]
m = mlp.Mlp(0.1)
epo = m.train(matTrain)

print("Terminou treinamento",epo)

print("test",m.test([0,1]))
print("test",m.test([1,1]))
print("test",m.test([1,0]))
print("test",m.test([0,0]))

