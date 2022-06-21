import numpy as np

#1. create the array
#list
np1=np.array([1,2,3,4],int)
print(np1,type(np1))
#[1 2 3 4] <class 'numpy.ndarray'>
#tuple
np2=np.array((5,6,7,8),float)
print(np2,type(np2))
#[5. 6. 7. 8.] <class 'numpy.ndarray'>

#2. arrange()
a=np.arange(0,31,2)
print(a)
#[ 0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30]

#3. linspace()
b=np.linspace(1,15,3)
print(b)
#[ 1.  8. 15.]

#4. zeros() and ones()
c=np.zeros((5,2))
print(c)
#[[0. 0.]
# [0. 0.]
# [0. 0.]
# [0. 0.]
# [0. 0.]]
d=np.ones((3,4))
print(d)
#[[1. 1. 1. 1.]
# [1. 1. 1. 1.]
# [1. 1. 1. 1.]]

#5. construct the N-dimensional matrix
listdata=[[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]]
e=np.array(listdata)
print(e)
print("維度",e.ndim)
print("形狀",e.shape)
print("數量",e.size)
#[[ 1  2  3  4  5]
# [ 6  7  8  9 10]
# [11 12 13 14 15]]
#維度 2
#形狀 (3, 5)
#數量 15

#6. change the shape of the array reshape()
f=np.arange(1,17)
print(f)
#[ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16]
g=f.reshape(4,4)
print(g)
#[[ 1  2  3  4]
# [ 5  6  7  8]
# [ 9 10 11 12]
# [13 14 15 16]]