import neuronio

n3 = neuronio.Neuronio(1,[1,1],0,[0.5,0.4],0.8)
print("fim",n3.y)
n3.ativar()
print("atic",n3.y)

n4 = neuronio.Neuronio(2,[1,1],0,[0.9,1],-0.1)
n4.ativar()
print("neuronio2",n4.y)