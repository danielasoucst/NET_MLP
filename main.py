import mlp

m = mlp.Mlp([1,1],0,0.1)
print(m.matWeight)
print("epocas:",m.train())
print("saida: ",m.test([0,1]))


