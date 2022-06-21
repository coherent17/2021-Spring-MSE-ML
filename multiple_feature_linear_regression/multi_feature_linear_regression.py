import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data=genfromtxt("multi_var.csv",delimiter=',')
x_data=data[:,0:2]
y_data=data[:,2]

#hypothesis function
def hypothesis(theta_0,theta_1,theta_2,x1,x2):
    return theta_0+theta_1*x1+theta_2*x2

#cost function
def square_error(theta_0,theta_1,theta_2,x_data,y_data):
    sqr_error=0
    for i in range(0,len(x_data)):
        sqr_error+=(hypothesis(theta_0,theta_1,theta_2,x_data[i,0],x_data[i,1])-y_data[i])**2
    return sqr_error/len(x_data)/2

#gradient descent
def gradient_descent(x_data,y_data,theta_0,theta_1,theta_2,learning_rate,iteration):
    N=len(x_data)
    cost_function=[]
    for i in range(iteration):
        theta_0_grad=0
        theta_1_grad=0
        theta_2_grad=0
        for j in range(0,N):
            theta_0_grad+=(1/N)*(hypothesis(theta_0,theta_1,theta_2,x_data[j,0],x_data[j,1])-y_data[j])
            theta_1_grad+=(1/N)*(hypothesis(theta_0,theta_1,theta_2,x_data[j,0],x_data[j,1])-y_data[j])*x_data[j,0]
            theta_2_grad+=(1/N)*(hypothesis(theta_0,theta_1,theta_2,x_data[j,0],x_data[j,1])-y_data[j])*x_data[j,1]
        theta_0=theta_0-(learning_rate*theta_0_grad)
        theta_1=theta_1-(learning_rate*theta_1_grad)
        theta_2=theta_2-(learning_rate*theta_2_grad)
        cost_function.append(square_error(theta_0,theta_1,theta_2,x_data,y_data))
        if i%200==0:
            #plot data
            ax=plt.figure().add_subplot(111,projection='3d')
            ax.scatter(x_data[:,0],x_data[:,1],y_data,c='r',marker='o',s=100)
            #plot the hypothesis
            x0=x_data[:,0]
            x1=x_data[:,1]
            x0,x1=np.meshgrid(x0,x1)
            z=theta_0+x0*theta_1+x1*theta_2
            ax.plot_surface(x0,x1,z)

            ax.set_title("the %d times of iteration" %(i))
            ax.set_xlabel("feature 1")
            ax.set_ylabel("feature 2")
            ax.set_zlabel("target")
            plt.show()
    return theta_0,theta_1,theta_2,cost_function

learning_rate=0.000001
theta_0=0
theta_1=0
theta_2=0
iteration=1000

print("Initial State:")
print("theta_0=%.3lf,   theta_1=%.3lf,  theta_2=%.3lf" %(theta_0,theta_1,theta_2))
print("Running for the method of gradient descent")
theta_0,theta_1,theta_2,cost_function=gradient_descent(x_data,y_data,theta_0,theta_1,theta_2,learning_rate,iteration)
print("Final state:")
print("theta_0=%.3lf,   theta_1=%.3lf,  theta_2=%.3lf" %(theta_0,theta_1,theta_2))

#plot the cost function versus iteration times
for i in range(0,len(cost_function)):
    plt.plot(i,cost_function[i],'b.')
plt.title("cost function versus iteration times")
plt.xlabel("iteration times")
plt.ylabel("cost function")
plt.show()