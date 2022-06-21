import numpy as np

A=np.array([[1,-0.1]
           ,[1,-0.15]])

B=np.array([np.log(70),np.log(60)])

C=np.linalg.solve(A,B)

print(C)


print(np.exp(4.5567966))