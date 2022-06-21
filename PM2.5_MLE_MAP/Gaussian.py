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

#shuffle the data to avoid the strange distribution
#concatenate the feature and target matrix and shuffle together
def shuffle(dataX,dataT):
    data_temp=np.c_[dataT,dataX]
    # np.random.shuffle(data_temp)
    dataT=data_temp[:,0]
    dataX=np.delete(data_temp,[0],axis=1)
    return dataX,dataT

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

def Gaussian(X):
    mean=[]
    std=[]
    for i in range(0,17):
        mean.append(np.mean(X[:,i]))
        std.append(np.std(X[:,i]))
    dataX_g=np.zeros(np.shape(X))
    for i in range(0,len(X)):
        for j in range(0,17):
            dataX_g[i,j]=(1/(std[j]*math.sqrt(2*2*math.pi)))*math.exp(-(X[i,j]-mean[j])**2/(2*std[j]**2))
    return dataX_g

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
dataX,dataT=shuffle(dataX,dataT)
dataX_g=Gaussian(dataX)
temp=np.ones((len(dataX_g),1))
dataX_g=np.c_[temp,dataX_g]

X_train,X_test,T_train,T_test = train_test_split(dataX_g,dataT,0.2)
w=linear_regression(X_train,T_train)

#plot the value of the model predict and the actual model (train part)
x=np.arange(0,len(X_train))
y=hypothesis(w.reshape(1,18),X_train).reshape(len(X_train),)
T_train=T_train.reshape(len(X_train),)
plt.plot(x,y,color='red',lw=1.0,ls='-',label="training_predict_value")
plt.plot(x,T_train,color='blue',lw=1.0,ls='-',label="target_value")
plt.text(0,1,"RMSE=%.3lf" %(rmse(T_train,y)))
plt.xlabel("the nth data")
plt.ylabel("PM2.5")
plt.title("Gaussian training")
plt.legend()
plt.show()

#plot the value of the model predict and the actual model (test part)
x=np.arange(0,len(X_test))
y=hypothesis(w.reshape(1,18),X_test).reshape(len(X_test),)
T_test=T_test.reshape(len(X_test),)
plt.plot(x,y,color='red',lw=1.0,ls='-',label="testing_predict_value")
plt.plot(x,T_test,color='blue',lw=1.0,ls='-',label="target_value")
plt.text(0,1,"RMSE=%.3lf" %(rmse(T_test,y)))
plt.xlabel("the nth data")
plt.ylabel("PM2.5")
plt.title("Gaussian testing")
plt.legend()
plt.show()

#plot the weight of each features
x=np.arange(0,18).reshape(18,)
plt.plot(x,w,'b.')
labels=["const","AMB_TEMP","CH4","CO","NMHC","NO","NO2","NOx","O3","PM10","RAINFALL","RH","SO2","THC","WD_HR","WIND_DIREC","WIND_SPEED","WS_HR"]
plt.xticks(x,labels,rotation='vertical')
plt.ylabel("weight")
plt.title("weights versus features(Gaussian)")
plt.show()

#remove each feature will how to affect the model?
rmse_remove=[]
for i in range(1,18):
    X_train_temp=np.delete(X_train,[i],axis=1)
    w_temp=linear_regression(X_train_temp,T_train)
    y_temp=hypothesis(w_temp.reshape(1,17),X_train_temp).reshape(len(X_train_temp),)
    rmse_remove.append(rmse(T_train,y_temp))
#plot the graph
x=np.arange(0,17)
y=hypothesis(w.reshape(1,18),X_train).reshape(len(X_train),)
rmse_org=[rmse(T_train,y)]*17
plt.plot(x,rmse_remove,"r.",label='The rmse without X feature')
plt.plot(x,rmse_org,'b.',label='rmse with all features')
labels=["AMB_TEMP","CH4","CO","NMHC","NO","NO2","NOx","O3","PM10","RAINFALL","RH","SO2","THC","WD_HR","WIND_DIREC","WIND_SPEED","WS_HR"]
plt.xticks(x,labels,rotation='vertical')
plt.ylabel("rmse")
plt.title("Linear regresion(Gaussian)--the impact on remove one feature")
plt.legend()
plt.show()