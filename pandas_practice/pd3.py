import pandas as pd

#1. sort by columns or by index
se1=pd.Series({'a':1,'b':6,'c':16})
se2=pd.Series({'a':2,'b':1,'c':5})
se3=pd.Series({'a':4,'b':8,'c':13})
se4=pd.Series({'a':9,'b':2,'c':17})
se5=pd.Series({'a':5,'b':10,'c':14})
e=pd.concat([se1,se2,se3,se4,se5],axis=1)
e.columns=['A','B','C','D','E']
print(e)
#    A  B   C   D   E
#a   1  2   4   9   5
#b   6  1   8   2  10
#c  16  5  13  17  14

#sort by columns
print(e.sort_values(by='D',ascending=False))
#D欄由大到小排列
#    A  B   C   D   E
#c  16  5  13  17  14
#a   1  2   4   9   5
#b   6  1   8   2  10

#sort by index
print(e.sort_index(axis=1))
#    A  B   C   D   E
#a   1  2   4   9   5
#b   6  1   8   2  10
#c  16  5  13  17  14

#change the certain value
e.loc['a']['C']=1000
print(e)
#    A  B     C   D   E
#a   1  2  1000   9   5
#b   6  1     8   2  10
#c  16  5    13  17  14

#del the certain data
print(e.drop('b'))
#    A  B     C   D   E
#a   1  2  1000   9   5
#c  16  5    13  17  14

print(e.drop('C',axis=1))
#    A  B   D   E
#a   1  2   9   5
#b   6  1   2  10
#c  16  5  17  14

print(e.drop(e.index[1:3]))
#   A  B     C  D  E
#a  1  2  1000  9  5

print(e.drop(e.columns[3:5],axis=1))
#    A  B     C
#a   1  2  1000
#b   6  1     8
#c  16  5    13