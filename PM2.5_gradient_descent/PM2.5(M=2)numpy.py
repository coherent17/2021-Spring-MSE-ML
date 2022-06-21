#M=2
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
    np.random.shuffle(data_temp)
    dataT=data_temp[:,0]
    dataX=np.delete(data_temp,[0],axis=1)
    return dataX,dataT

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

# append the dataX to match the theta (171 features)
def data_preprocessing(dataX):
    k=18
    for i in range(1,18):
        for j in range(1,i+1):
            if k in range(18,171):
                dataX=np.insert(dataX,k,values=dataX[:,i]*dataX[:,j],axis=1)
                k+=1
    return dataX

#split the data into training set and the testing set
def train_test_split(X,Y,test_size):
    X_train=np.array(X[:math.floor(len(X)*(1-test_size))])
    Y_train=np.array(Y[:math.floor(len(Y)*(1-test_size))])
    X_test=np.array(X[math.floor(len(X)*(1-test_size)):])
    Y_test=np.array(Y[math.floor(len(Y)*(1-test_size)):])
    Y_train=Y_train.reshape(1,len(Y_train))
    Y_test=Y_test.reshape(1,len(Y_test))
    return X_train, X_test, Y_train, Y_test

#hypothesis function
def hypothesis(theta,X):
    return np.matmul(theta,np.transpose(X))

#gradient descent
def gradient_descent(theta,X,T,learning_rate,iteration):
    N=len(X)
    cost_function=[]
    for i in range(1,iteration+1):
        cost_function.append(np.sum((hypothesis(theta,X)-T)**2)/len(X)/2)
        theta_grad=(1/N)*np.matmul((hypothesis(theta,X)-T),(X))
        theta-=learning_rate*theta_grad
    return theta,cost_function

#root mean square error
def rmse(a,b):
    return math.sqrt(np.sum((a-b)**2)/len(a))

#parameter:
learning_rate=0.01
iteration=10000
theta=np.zeros((1,171))

dataX,dataT=normalization(dataX,dataT)
dataX,dataT=shuffle(dataX,dataT)
temp=np.array([1]*len(dataX))
dataX=np.c_[temp,dataX]

# append the dataX to match the theta (171 features)
dataX=data_preprocessing(dataX)
X_train,X_test,T_train,T_test = train_test_split(dataX,dataT, test_size = 0.2)
theta,cost_function=gradient_descent(theta,X_train,T_train,learning_rate,iteration)

#plot the cost function versus iteration times
x=np.arange(0,len(cost_function))
plt.plot(x,cost_function,'b.')
plt.title("cost function versus iteration times")
plt.xlabel("iteration times")
plt.ylabel("cost function")
plt.show()

#plot the value of the model predict and the actual model (train part)
x=np.arange(0,len(X_train))
y=hypothesis(theta,X_train).reshape(len(X_train),)
T_train=T_train.reshape(len(X_train),)
plt.plot(x,y,color='red',lw=1.0,ls='-',label="training_predict_value")
plt.plot(x,T_train,color='blue',lw=1.0,ls='-',label="target_value")
plt.text(0,1,"RMSE=%.3lf" %(rmse(T_train,y)))
plt.xlabel("the nth data")
plt.ylabel("PM2.5")
plt.title("Linear regression (M=2) training")
plt.legend()
plt.show()

#plot the value of the model predict and the actual model (test part)
x=np.arange(0,len(X_test))
y=hypothesis(theta,X_test).reshape(len(X_test),)
T_test=T_test.reshape(len(X_test),)
plt.plot(x,y,color='red',lw=1.0,ls='-',label="testing_predict_value")
plt.plot(x,T_test,color='blue',lw=1.0,ls='-',label="target_value")
plt.text(0,1,"RMSE=%.3lf" %(rmse(T_test,y)))
plt.xlabel("the nth data")
plt.ylabel("PM2.5")
plt.title("Linear regression (M=2) testing")
plt.legend()
plt.show()