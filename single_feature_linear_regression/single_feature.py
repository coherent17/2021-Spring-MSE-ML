import numpy as np
import matplotlib.pyplot as plt

data=np.genfromtxt("data.txt",delimiter=",")
x_data=data[:,0]
y_data=data[:,1]
# plt.plot(x_data,y_data,'b.',markersize=13)
# plt.show()

#hypothesis function
def hypothesis(theta_0,theta_1,x):
    return theta_0+theta_1*x

#cost function
def square_error(theta_0,theta_1,x_data,y_data):
    sqr_error=0
    for i in range(0,len(x_data)):
        #the difference between hypothesis function and the correct value
        sqr_error+=((hypothesis(theta_0,theta_1,x_data[i])-y_data[i]))**2
    return sqr_error/len(x_data)/2

#gradient descent
def gradient_descent(x_data,y_data,theta_0,theta_1,learning_rate,iteration):
    N=len(x_data)
    #store the value of each iteration in cost function
    cost_function=[]
    for i in range(iteration):
        theta_0_grad=0
        theta_1_grad=0
        for j in range(0,N):
            theta_0_grad+=(1/N)*(hypothesis(theta_0,theta_1,x_data[j])-y_data[j])
            theta_1_grad+=(1/N)*(hypothesis(theta_0,theta_1,x_data[j])-y_data[j])*x_data[j]
        theta_0=theta_0-(learning_rate*theta_0_grad)
        theta_1=theta_1-(learning_rate*theta_1_grad)
        cost_function.append(square_error(theta_0,theta_1,x_data,y_data))
        if i%50==0:
            plt.plot(x_data,y_data,'b.')
            plt.plot(x_data,theta_1*x_data+theta_0,'r')
            plt.title("the %d times of iteration" %(i))
            plt.xlabel("x data")
            plt.ylabel("y data")
            plt.show()
    return theta_0,theta_1,cost_function

learning_rate=0.00001
theta_0=0
theta_1=0
iteration=200


print("Initial state:")
print("theta_0=%.3lf,   theta_1=%.3lf" %(theta_0,theta_1))
print("Running for the method of gradient descent")
theta_0,theta_1,cost_function=gradient_descent(x_data,y_data,theta_0,theta_1,learning_rate,iteration)
print("Final state:")
print("theta_0=%.3lf,   theta_1=%.3lf" %(theta_0,theta_1))

# plt.plot(x_data,y_data,'b.')
# plt.plot(x_data,theta_1*x_data+theta_0,'r')
# plt.show()

for i in range(0,len(cost_function)):
    plt.plot(i,cost_function[i],'b.')
plt.title("cost function versus iteration times")
plt.xlabel("iteration times")
plt.ylabel("cost function")
plt.show()