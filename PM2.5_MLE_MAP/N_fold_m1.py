import numpy as np
import matplotlib.pyplot as plt
import math 
import random

#data preprocesing:
#feature data
dataX=np.genfromtxt("dataset_X.csv",delimiter=',')
dataX=np.delete(dataX,[0],axis=1)

#target data
dataT=np.genfromtxt("dataset_T.csv",delimiter=',')
dataT=np.delete(dataT,[0],axis=1)

#normalize:
def normalization(dataX,dataT):
    #features
    mean_X=[]
    std_X=[]
    for i in range(0,17):
        mean_X.append(np.mean(dataX[:,i]))
        std_X.append(np.std(dataX[:,i]))

    dataX_n=np.zeros(np.shape(dataX))
    for i in range(0,len(dataX)):
        for j in range(0,17):
            dataX_n[i,j]=(dataX[i,j]-mean_X[j])/std_X[j]
    dataX=dataX_n
    #target
    mean_T=np.mean(dataT[:])
    std_T=np.std(dataT[:])
    dataT_n=np.zeros(np.shape(dataT))
    for i in range(0,len(dataT)):
        dataT_n[i]=(dataT[i]-mean_T)/std_T
    return dataX,dataT

#shuffle the data to avoid the strange distribution
#concatenate the feature and target matrix and shuffle together
def shuffle(dataX,dataT):
    data_temp=np.c_[dataT,dataX]
    np.random.shuffle(data_temp)
    dataT=data_temp[:,0].reshape(1096,1)
    dataX=np.delete(data_temp,[0],axis=1)
    return dataX,dataT

def train_test_split(X,Y,test_size):
    X_train=np.array(X[:math.floor(len(X)*(1-test_size))])
    Y_train=np.array(Y[:math.floor(len(Y)*(1-test_size))])
    X_test=np.array(X[math.floor(len(X)*(1-test_size)):])
    Y_test=np.array(Y[math.floor(len(Y)*(1-test_size)):])
    Y_train=Y_train.reshape(len(Y_train),1)
    Y_test=Y_test.reshape(len(Y_test),1)
    return X_train, X_test, Y_train, Y_test

def linear_regression(X,Y):
    w=np.matmul(np.matmul(np.linalg.pinv(np.matmul(X.T,X)),X.T),Y)
    return w

def hypothesis(w,X):
    return np.matmul(w,np.transpose(X))

def rmse(a,b):
    return math.sqrt(np.sum((a-b)**2)/len(a))

dataX,dataT=normalization(dataX,dataT)
temp=np.array([1]*len(dataX))
dataX=np.c_[temp,dataX]
dataX,dataT=shuffle(dataX,dataT)

#N_fold cross validation
rmse_org=[]
rmse_D5=[]
rmse_D12=[]
for i in range(10):
    #original part: 
    X_train,X_test,T_train,T_test=train_test_split(dataX,dataT,0.1)

    #get the D5 training and testing dataset
    X_train_D5=np.delete(X_train,[1,2,4,5,6,7,8,12,13,14,15,16],axis=1)
    X_test_D5=np.delete(X_test,[1,2,4,5,6,7,8,12,13,14,15,16],axis=1)

    #get the D12 training and testing dataset
    X_train_D12=np.delete(X_train,[1,5,6,7,8],axis=1)
    X_test_D12=np.delete(X_test,[1,5,6,7,8],axis=1)

    #original linear regression part
    w_org=linear_regression(X_train,T_train)
    y_org=hypothesis(w_org.reshape(1,18),X_test).reshape(len(X_test),)
    T_test=T_test.reshape(len(X_test),)
    rmse_org.append(rmse(T_test,y_org))

    #D5 linear regression part
    w_D5=linear_regression(X_train_D5,T_train)
    y_D5=hypothesis(w_D5.reshape(1,6),X_test_D5).reshape(len(X_test_D5),)
    rmse_D5.append(rmse(T_test,y_D5))

    #D12 linear regression part
    w_D12=linear_regression(X_train_D12,T_train)
    y_D12=hypothesis(w_D12.reshape(1,13),X_test_D12).reshape(len(X_test_D12),)
    rmse_D12.append(rmse(T_test,y_D12))

    #update the data
    ind=np.arange(math.floor(len(dataX)*(1-0.1)),np.shape(dataX)[0])
    dataX=np.delete(dataX,ind,axis=0)
    dataX=np.c_[X_test.T,dataX.T].T
    dataT=np.delete(dataT,ind,axis=0)
    T_test=T_test.reshape(110,1)
    dataT=np.vstack((T_test,dataT))
    
print(rmse_org)
print(rmse_D5)
print(rmse_D12)

x=np.arange(0,10)
plt.plot(x,rmse_org,label="rmse org")
plt.plot(x,rmse_D5,label="rmse D5")
plt.plot(x,rmse_D12,label="rmse D12")
plt.title("linear model by different dimension of feature (m1)")
plt.xlabel("N-fold")
plt.ylabel("rmse")
plt.legend()
plt.show()