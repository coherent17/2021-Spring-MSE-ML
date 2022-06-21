import pandas as pd

#1. create the series by pandas
a=pd.Series([1,2,3,4,5])
print(a)
#0    1
#1    2
#2    3
#3    4
#4    5
#dtype: int64
print(a.values)
#[1 2 3 4 5]
print(a.index)
#RangeIndex(start=0, stop=5, step=1)
print(a[2]) #3
print(a[2:5])
#2    3
#3    4
#4    5
#dtype: int64

#create the own index
b=pd.Series([1,2,3,4,5],index=['a','b','c','d','e'])
print(b)
#a    1
#b    2
#c    3
#d    4
#e    5
#dtype: int64
print(b['b']) #2

#use dictionary to create series
dict1={'NCTU':'交大','NTHU':'清大','NTU':'台大'}
c=pd.Series(dict1)
print(c)
#NCTU    交大
#NTHU    清大
#NTU     台大
#dtype: object
print(c.values)
#['交大' '清大' '台大']
print(c.index)
#Index(['NCTU', 'NTHU', 'NTU'], dtype='object')
print(c['NCTU']) #交大