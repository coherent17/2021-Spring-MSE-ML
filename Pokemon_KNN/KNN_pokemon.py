import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

class KNN():
    def __init__(self):
        pass

    def data_preprocessing(self):
        dataX=np.genfromtxt("Pokemon_feature.csv",delimiter=',',dtype=str)
        for i in range(len(dataX)):
            if dataX[i,-1]=='FALSE':
                dataX[i,-1]='0'
            else:
                dataX[i,-1]='1'

        dataX_int=np.zeros((dataX.shape[0],dataX.shape[1]),dtype=float)
        for i in range(dataX.shape[0]):
            for j in range(dataX.shape[1]):
                dataX_int[i,j]=float(dataX[i,j])

        dataX=dataX_int

        #target water=1, normal=2, psychic=3
        dataT=np.genfromtxt("Pokemon_target.csv",delimiter=',',dtype=str)
        dataT_int=np.zeros(len(dataT),dtype=float)
        for i in range(0,len(dataT)):
            if i==0: #read file error
                dataT_int[i]=2
            elif dataT[i]=='Water':
                dataT_int[i]=1
            elif dataT[i]=='Normal':
                dataT_int[i]=2
            elif dataT[i]=='Psychic':
                dataT_int[i]=3

        dataT=dataT_int

        return dataX,dataT

    def normalize(self,X):
        mean_X=[]
        std_X=[]
        for i in range(X.shape[1]):
            mean_X.append(np.mean(X[:,i]))
            std_X.append(np.std(X[:,i]))
        X_n=np.zeros(np.shape(X))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X_n[i,j]=(X[i,j]-mean_X[j])/std_X[j]
        return X_n

    def train_test_split(self,dataX,dataT):
        dataX_train=dataX[0:120,:] #(120,9))
        dataX_test=dataX[120:,:] #(38,9)
        dataT_train=dataT[0:120].reshape(120,1) #(120,1)
        dataT_test=dataT[120:].reshape(38,1) #(38,1)
        return dataX_train,dataX_test,dataT_train,dataT_test

    #per row of the testing data return a list of the euclidean distance
    def euclidean_distance(self,X_train,X_test):
        distance=[]
        for i in range(np.shape(X_train)[0]):
            dis=0
            for j in range(np.shape(X_train)[1]):
                dis+=(X_train[i,j]-X_test[j])**2
            distance.append(np.sqrt(dis))
        return distance

    def vote(self,distance,k,dataT_train):
        ind_sort=np.argsort(distance)
        ind=[]
        for i in range(k):
            ind.append(ind_sort[i])
        flag=[0,0,0]
        for i in dataT_train[ind]:
            if i==1: # water
                flag[0]+=1
            elif i==2: # normal
                flag[1]+=1
            elif i==3: # psychic
                flag[2]+=1
        big=np.max(flag)
        result=[]
        if flag[0]==big:
            result='water'
        elif flag[1]==big:
            result='normal'
        elif flag[2]==big:
            result='psychic'
        return result

    def accuracy(self,result,dataT_test):
        flag=0
        for i in range(len(result)):
            if result[i]=='normal' and dataT_test[i]==2:
                flag+=1
            elif result[i]=='water' and dataT_test[i]==1:
                flag+=1
            elif result[i]=='psychic' and dataT_test[i]==3:
                flag+=1
        return flag/len(result)

    def KNN_score(self,dataX_train,dataX_test,dataT_train,dataT_test):
        accuracy_store=[] #store the accuracy of different K from 1 ~ 10
        for j in range(1,11): #change the value of K
            result=[]
            for i in range(np.shape(dataX_test)[0]): #change the row of the testing data
                distance=KNN.euclidean_distance(dataX_train, dataX_test[i,:])
                result.append(KNN.vote(distance,j,dataT_train))
            accuracy_store.append(KNN.accuracy(result, dataT_test))
        k=np.argsort(accuracy_store)[-1] #the K value corresponding to the highest accuracy
        print('the maximum accuracy happen at K= %d' %(k+1))
        return accuracy_store,k

    def KNN_classfier(self,index,dataX_train,dataX_test,dataT_train,dataT_test,k): #use the highest k to classfier the data
        distance=KNN.euclidean_distance(dataX_train, dataX_test[0,:])
        result=(KNN.vote(distance,k,dataT_train))
        print('the prediction type: '+result)
        if result=='water' and dataT_test[index]==1:
            print('the prediction is correct!')
        elif result=='normal' and dataT_test[index]==2:
            print('the prediction is correct!')
        elif result=='psychic' and dataT_test[index]==3:
            print('the prediction is correct!')
        else:
            print('the prediction is wrong!')
            if dataT_test[index]==1:
                print('the correct answer is water')
            elif dataT_test[index]==2:
                print('the correct answer is normal')
            elif dataT_test[index]==3:
                print('the correct answer is psychic')
        return result

KNN=KNN()
dataX,dataT=KNN.data_preprocessing()
dataX_train,dataX_test,dataT_train,dataT_test=KNN.train_test_split(dataX, dataT)
dataX_train=KNN.normalize(dataX_train)
dataX_test=KNN.normalize(dataX_test)
accuracy_store,k=KNN.KNN_score(dataX_train, dataX_test, dataT_train, dataT_test)
a=np.random.randint(np.shape(dataX_test)[0])
result=KNN.KNN_classfier(a,dataX_train, dataX_test[a,:].reshape(1,9), dataT_train, dataT_test,k)

#visualize
x=np.linspace(1, 10,10)
plt.plot(x,accuracy_store)
plt.xlabel('K')
plt.ylabel('accuracy')
plt.title('accuracy versus the value of K')
plt.grid(True)
plt.show()