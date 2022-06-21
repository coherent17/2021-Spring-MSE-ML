import pandas as pd

#input and output the data
data=pd.read_csv('test.csv',header=0,index_col=0)
print(data)
print(type(data))
data.to_csv('testout.csv',encoding='utf-8-sig')