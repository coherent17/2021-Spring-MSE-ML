import pandas as pd
import matplotlib.pyplot as plt 

#data visualization
e=pd.DataFrame([[250,320,300,312,280],
                [280,300,280,290,310],
                [220,280,250,305,250]],
                index=['Taipei','Taichung','Tainan'],
                columns=[2015,2016,2017,2018,2019])
print(e)
#          2015  2016  2017  2018  2019
#Taipei     250   320   300   312   280
#Taichung   280   300   280   290   310
#Tainan     220   280   250   305   250
graph1=e.plot(kind='bar',title='bar',figsize=[10,5])
graph2=e.plot(kind='barh',title='barh',figsize=[10,5])
graph3=e.plot(kind='bar',title='stack',stacked=True,figsize=[10,5])
plt.show()
graph4=e.iloc[0].plot(kind='line',legend=True,figsize=[10,5],xticks=range(2015,2020))
graph4=e.iloc[1].plot(kind='line',legend=True,figsize=[10,5],xticks=range(2015,2020))
graph4=e.iloc[2].plot(kind='line',legend=True,figsize=[10,5],xticks=range(2015,2020))
plt.show()
graph5=e.plot(kind='pie',subplots='True',figsize=[20,20])
plt.show()
