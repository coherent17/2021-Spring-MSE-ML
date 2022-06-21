import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

boston_dataset = load_boston()
#get the boston data from the sklearn.dataset and assign to the boston_dataset (Dict)
#in this dictionary include ['data', 'target', 'feature_names', 'DESCR', 'filename']
#data:contains the information for various houses
#target: prices of the house
#feature_names: names of the features
#DESCR: describes the dataset

#in boston_dataset.DESCR describe the variable in the boston_dataset.data but without the MEDV
# so we need to get the MEDV in the target and assign into the boston

boston=pd.DataFrame(boston_dataset.data,columns=boston_dataset.feature_names)
#append a index in boston to storage the MEDV from boston_dataset.target
boston['MEDV']=boston_dataset.target

#data visualize
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(boston['MEDV'], bins=30)
#hist and kde
plt.show()

#build a correlation matrix to measure the linear relationships between the variables in the data
correlation_matrix = boston.corr().round(2)
#get the correlation value between the boston and round to the second decimal place
sns.heatmap(data=correlation_matrix, annot=True)
#output by the heatmap
plt.show()

#correlation matrix:
#         CRIM    ZN  INDUS  CHAS   NOX    RM   AGE   DIS   RAD   TAX  PTRATIO     B  LSTAT  MEDV
#CRIM     1.00 -0.20   0.41 -0.06  0.42 -0.22  0.35 -0.38  0.63  0.58     0.29 -0.39   0.46 -0.39
#ZN      -0.20  1.00  -0.53 -0.04 -0.52  0.31 -0.57  0.66 -0.31 -0.31    -0.39  0.18  -0.41  0.36
#INDUS    0.41 -0.53   1.00  0.06  0.76 -0.39  0.64 -0.71  0.60  0.72     0.38 -0.36   0.60 -0.48
#CHAS    -0.06 -0.04   0.06  1.00  0.09  0.09  0.09 -0.10 -0.01 -0.04    -0.12  0.05  -0.05  0.18
#NOX      0.42 -0.52   0.76  0.09  1.00 -0.30  0.73 -0.77  0.61  0.67     0.19 -0.38   0.59 -0.43
#RM      -0.22  0.31  -0.39  0.09 -0.30  1.00 -0.24  0.21 -0.21 -0.29    -0.36  0.13  -0.61  0.70
#AGE      0.35 -0.57   0.64  0.09  0.73 -0.24  1.00 -0.75  0.46  0.51     0.26 -0.27   0.60 -0.38
#DIS     -0.38  0.66  -0.71 -0.10 -0.77  0.21 -0.75  1.00 -0.49 -0.53    -0.23  0.29  -0.50  0.25
#RAD      0.63 -0.31   0.60 -0.01  0.61 -0.21  0.46 -0.49  1.00  0.91     0.46 -0.44   0.49 -0.38
#TAX      0.58 -0.31   0.72 -0.04  0.67 -0.29  0.51 -0.53  0.91  1.00     0.46 -0.44   0.54 -0.47
#PTRATIO  0.29 -0.39   0.38 -0.12  0.19 -0.36  0.26 -0.23  0.46  0.46     1.00 -0.18   0.37 -0.51
#B       -0.39  0.18  -0.36  0.05 -0.38  0.13 -0.27  0.29 -0.44 -0.44    -0.18  1.00  -0.37  0.33
#LSTAT    0.46 -0.41   0.60 -0.05  0.59 -0.61  0.60 -0.50  0.49  0.54     0.37 -0.37   1.00 -0.74
#MEDV    -0.39  0.36  -0.48  0.18 -0.43  0.70 -0.38  0.25 -0.38 -0.47    -0.51  0.33  -0.74  1.00
#by choosing the data which is correlated to the MEDV, we discover that RM and LSTAT has most positive and negative correlaton

#using scatter plot to visualize the correlation between MEDV and (RM or LSTAT)
plt.figure(figsize=(20, 5))
features = ['LSTAT', 'RM']
target = boston['MEDV']
for i, col in enumerate(features):
    plt.subplot(1,2,i+1)
    x = boston[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')
plt.show()

#prepare the data for training the model
X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns = ['LSTAT','RM'])
Y = boston['MEDV']
# Splitting the data into training and testing sets
#80% for training set and 20% for testing set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
#X_train:[404 rows x 2 columns]
#X_test:[102 rows x 2 columns]
#Y_train:Name: MEDV, Length: 404, dtype: float64
#Y_test:Name: MEDV, Length: 102, dtype: float64

#Training and testing the model
lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)

# model evaluation for training set
y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))