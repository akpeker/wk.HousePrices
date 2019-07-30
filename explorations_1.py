# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 19:59:46 2019

@author: PEKERPCLocal
"""

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

train = pd.read_csv(r'data\train.csv')
test = pd.read_csv(r'data\test.csv')

plt.hist(train.SalePrice,bins=40)
plt.hist(np.log(train.SalePrice),bins=30)

train.SalePrice.skew()
np.log(train.SalePrice).skew()
target = np.log(train.SalePrice)

# Select a subset of columns based on dtype (in this case, numeric)
numeric_features = train.select_dtypes(include=[np.number])
numeric_features.dtypes

c0 = train.corr()
#c0 = numeric_features.corr()
c0['SalePrice'].sort_values(ascending=False)[:10]
c0['SalePrice'].sort_values(ascending=False)[-5:]

plt.plot(c0.SalePrice.sort_values(),"o-")
plt.xticks(rotation=90)

train.OverallQual.unique()

sns.boxplot(x="Heating",y="logSP",data=train)
plt.figure()
sns.distplot(train.SalePrice)

def uniq():
    for c in train.columns:
        try:
            x = sorted(train[c].unique())
        except:
            x = train[c].unique()
        print("-"*40,"\n",c,len(x))
        if len(x) < 30:
            print(x)
        else:
            print("%d values. Min=%s, Max=%s" % (len(x), x[0], x[-1]))

def PivT():
    quality_pivot = train.pivot_table(index='OverallQual',
                      values='SalePrice', aggfunc=np.median)
    quality_pivot.plot(kind='bar', color='blue')
    plt.xlabel('Overall Quality')
    plt.ylabel('Median Sale Price')
    plt.xticks(rotation=0)
  
def Scat():
    plt.scatter(x=train['GrLivArea'], y=target)
    plt.ylabel('Sale Price')
    plt.xlabel('Above grade (ground) living area square feet')

    plt.scatter(x=train['GarageArea'], y=target)
    plt.ylabel('Sale Price')
    plt.xlabel('Garage Area')

  