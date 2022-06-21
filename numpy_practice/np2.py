import numpy as np

#1. get the value of the one dimensional array
a=np.arange(0,6)   
print(a)                #[0 1 2 3 4 5]       
print(a[0])             #0
print(a[5])             #5
print(a[1:5])           #[0 1 2 3 4]
print(a[1:5:2])         #[1 3]
print(a[:])             #[0 1 2 3 4 5]
print(a[:3])            #[0 1 2]
print(a[3:])            #[3 4 5]

#2. get the value of the N dimensional array
b=np.arange(1,17).reshape(4,4)
print(b)
#[[ 1  2  3  4]
# [ 5  6  7  8]
# [ 9 10 11 12]
# [13 14 15 16]]
print(b[2,3])           #12
print(b[1,1:3])         #[6 7]
print(b[1:3,2])         #[7 11]
print(b[1:3,1:3])       #[[6 7],[10 11]]

#3. random
#0~1之間的隨機浮點數
print(np.random.rand(2,3))

#常態分佈的隨機浮點數
print(np.random.randn(2,3))

#產生範圍內的隨機整數不包括上限,[size]
print(np.random.randint(0,5,[6]))

#產生0~42中,6個不重複的隨機整數
print(np.random.choice(43,6,replace=False))

#在陣列中隨機取6個可以重複的隨機整數
dataset=[0,5,6,3,2,4,8,6,5,6,9,7,3,11,12,15,16,19]
print(np.random.choice(dataset,6,replace=True))

#4. read the data please store the data outside the folder 
data=np.genfromtxt("score.csv", delimiter=",",skip_header=1)
print(data)