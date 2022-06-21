import numpy as np

#1. numpy calculation
a=np.arange(1,10).reshape(3,3)
b=np.arange(10,19).reshape(3,3)
print(a)
print(b)
#所有元素都加1
print(a+1)

#所有元素都平方
print(a**2)

#判斷式輸出boolen
print(a<5)

#相對應的元素相加
print(a*b)

#矩陣內積
print(np.dot(a,b))

#2. function i numpy and statistic
c=np.arange(1,10).reshape(3,3)
print(c)
#[[1 2 3]
# [4 5 6]
# [7 8 9]]

#最小值與最大值
print(np.min(c),np.max(c))                  #1 9

#總和 乘積 平均
print(np.sum(c),np.product(c),np.mean(c))   #45 362880 5.0

#標準差
print(np.std(c))                            #2.581988897471611

#變異數
print(np.var(c))                            #6.666666666666667

#中位數
print(np.median(c))                         #5.0

#max-min
print(np.ptp(c))                            #8

#3. sort in numpy
d=np.random.choice(50,size=10,replace=False)
#before sorting
print(d)
#after sorting
print(np.sort(d))
#the index after sorting
print(np.argsort(d))
#use the index to get the value
for i in np.argsort(d):
    print(d[i],end=" ")
print("\n")

#N dimension array sorting
e=np.random.randint(0,10,(3,5))
print(e)
#對每一直行sort
print(np.sort(e,axis=0))
#對每一橫列sort
print(np.sort(e,axis=1))