import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.datasets import load_boston

def train_test_split(X,Y,test_size):
    X_train=X[:math.floor(len(X)*(1-test_size))]
    Y_train=Y[:math.floor(len(Y)*(1-test_size))]
    X_test=X[math.floor(len(X)*(1-test_size)):]
    Y_test=Y[math.floor(len(Y)*(1-test_size)):]
    return X_train, X_test, Y_train, Y_test

def hypothesis(theta,X,n):
    #X.shape[0]:資料數
    h=np.ones((X.shape[0],1))
    theta=theta.reshape(1,n+1)
    for i in range(0,X.shape[0]):
        h[i]=float(np.matmul(theta,X[i]))
    h=h.reshape(X.shape[0])
    return h

def BGD(theta,learning_rate,num_iterations,h,X,y,n):
    cost=np.ones(num_iterations)
    for i in range(0,num_iterations):
        theta[0]=theta[0]-(alpha/X.shape[0])*sum(h-y)
        for j in range(1,n+1):
            theta[j]=theta[j]-(alpha/X.shape[0])*sum(h-y)*X.transpose([j])
        h=hypothesis(theta,X,n)
        cost[i]=(1/X.shape[0])*0.5*sum(np.square(h-y))
    theta=theta.reshape(1,n+1)
    return theta,cost

def linear_regression(X,y,alpha,num_iterations):
    n=X.shape[0]
    one_column=np.ones((X.shape[0],1))
    X=np.concatenate((one_column,X),axis=1)
    theta=np.zeros(n+1)
    h=hypothesis(theta,X,n)
    theta,cost=BGD(theta,alpha,num,num_iterations,h,X,y,n)
    return theta,cost

boston_dataset = load_boston()
boston=pd.DataFrame(boston_dataset.data,columns=boston_dataset.feature_names)
boston['MEDV']=boston_dataset.target

X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns = ['LSTAT','RM'])
Y = boston['MEDV']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

