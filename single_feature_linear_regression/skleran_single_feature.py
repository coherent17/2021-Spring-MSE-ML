from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

data=np.genfromtxt("data.txt",delimiter=",")

#the linear model need to reshape the data
x_data=data[:,0,np.newaxis]
y_data=data[:,1,np.newaxis]

model=LinearRegression()
model.fit(x_data,y_data)

plt.plot(x_data,y_data,'b.')
plt.plot(x_data,model.predict(x_data),'r')
plt.show()