import numpy as np
import matplotlib.pyplot as plt
import math

#read data
#feature data
dataX=np.genfromtxt("x.csv",delimiter=',')
#target data
dataT=np.genfromtxt("t.csv",delimiter=',')
dataT=dataT.reshape(len(dataT),1)

def normalization(X):
    M=3
    s=0.1
    x_1=(X-(2*0/M))/s
    x_2=(X-(2*1/M))/s
    x_3=(X-(2*2/M))/s
    return np.column_stack((x_1,x_2,x_3))

def sigmoidal(X):
    return 1/(1+np.exp(-1*X))

def bayesian_regression(X,T):
    beta=1
    S_0_inv=np.eye(3)*(10**-6)
    m_0=np.zeros((3,1))
    S_N_inv=S_0_inv+beta*(X.T@X)
    S_N=np.linalg.inv(S_N_inv)
    temp=(S_0_inv@m_0)
    m_N=(S_N@((S_0_inv@m_0)+beta*((X.T@T))))
    return m_N,S_N

N=80
dataX_nor=normalization(dataX[0:N])
dataX_s=sigmoidal(dataX_nor)
m_N,S_N=bayesian_regression(dataX_s,dataT[0:N])

#generate 5 sample curves
#the testing dataset need to pass same data preprocessing
x1=np.linspace(0,2,10000)
x1_nor=normalization(x1)
x1_s=sigmoidal(x1_nor)
weight=np.random.multivariate_normal(m_N.reshape(3,),S_N).reshape(3,1)
y1=x1_s@weight

x2=np.linspace(0,2,10000)
x2_nor=normalization(x2)
x2_s=sigmoidal(x2_nor)
weight=np.random.multivariate_normal(m_N.reshape(3,),S_N).reshape(3,1)
y2=x2_s@weight

x3=np.linspace(0,2,10000)
x3_nor=normalization(x3)
x3_s=sigmoidal(x3_nor)
weight=np.random.multivariate_normal(m_N.reshape(3,),S_N).reshape(3,1)
y3=x3_s@weight

x4=np.linspace(0,2,10000)
x4_nor=normalization(x4)
x4_s=sigmoidal(x4_nor)
weight=np.random.multivariate_normal(m_N.reshape(3,),S_N).reshape(3,1)
y4=x4_s@weight

x5=np.linspace(0,2,10000)
x5_nor=normalization(x5)
x5_s=sigmoidal(x5_nor)
weight=np.random.multivariate_normal(m_N.reshape(3,),S_N).reshape(3,1)
y5=x5_s@weight

#visualize
plt.plot(dataX[0:N],dataT[0:N],'bo',label='training data')
plt.plot(x1,y1,label='sample curve 1')
plt.plot(x2,y2,label='sample curve 2')
plt.plot(x3,y3,label='sample curve 3')
plt.plot(x4,y4,label='sample curve 4')
plt.plot(x5,y5,label='sample curve 5')
plt.xlabel('x')
plt.ylabel('t')
plt.title('training data size: %d'%(N))
plt.legend()
plt.show()

#mean and std calculation
sample_mean=[]
sample_std=[]
hbound=[]
lbound=[]
for i in range(len(y1)):
    sample_mean.append(np.mean([y1[i],y2[i],y3[i],y4[i],y5[i]]))
    sample_std.append(np.std([y1[i],y2[i],y3[i],y4[i],y5[i]]))
    hbound.append(sample_mean[i]+sample_std[i])
    lbound.append(sample_mean[i]-sample_std[i])

#visualize
plt.plot(dataX[0:N],dataT[0:N],'bo',label='training data')
plt.plot(x1,sample_mean,label='mean curve')
plt.fill_between(x1,sample_mean,hbound,facecolor='pink')
plt.fill_between(x1,sample_mean,lbound,facecolor='pink')
plt.xlabel('x')
plt.ylabel('t')
plt.title('training data size: %d'%(N))
plt.legend()
plt.show()

#sample weight calculation and scatter 
weight0=[]
weight1=[]
weight2=[]
for i in range(10000):
    weight=np.random.multivariate_normal(m_N.reshape(3,),S_N).reshape(3,1)
    weight0.append(weight[0])
    weight1.append(weight[1])
    weight2.append(weight[2])
weight0 = np.array(weight0).reshape(10000,)
weight1 = np.array(weight1).reshape(10000,)
weight2 = np.array(weight2).reshape(10000,)
plt.hist2d(np.array(weight0),np.array(weight1),range=[[0,5],[-6,6]],bins = 50,cmap = 'rainbow')
plt.xlabel("w[0]")
plt.ylabel("w[1]")
plt.title('weight scatter with training data size: %d' %(N))
plt.colorbar()
plt.show()