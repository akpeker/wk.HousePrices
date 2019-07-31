# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 19:59:46 2019

@author: PEKERPCLocal

Description:
        Snippets of code for basic exploration of data. Moslty inspired by
        couple of blog posts and tutorials.
"""

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

# Load data
#-----------------------------------------------------------------------------
train = pd.read_csv(r'data\train.csv')
test = pd.read_csv(r'data\test.csv')


# We will use log of SalePrice since the competition does that for error calculation.
#-----------------------------------------------------------------------------
target = np.log(train.SalePrice)
train["logSP"] = target


#-----------------------------------------------------------------------------
# Look at the distribution of the target. 
# Seaborn's plots are nice and are worth a look.
plt.figure()
plt.hist(train.SalePrice,bins=40)
plt.title("SalePrice distribution")
plt.figure()
plt.hist(train.logSP,bins=30)
plt.title("log(SalePrice) distribution")
# Do the same with seaborn
plt.figure()
sns.distplot(train.SalePrice)
plt.title("SalePrice distribution")
plt.figure()
sns.distplot(train.logSP,bins=30)
plt.title("log(SalePrice) distribution")


# Skew of the target distribution
#-----------------------------------------------------------------------------
print( "\n\n{}".format("="*80))
print( "SalePrice skew      = %f." % train.SalePrice.skew() )
print( "log(SalePrice) skew = %f." % np.log(train.SalePrice).skew() )


# Select a subset of columns based on dtype (in this case, numeric)
#-----------------------------------------------------------------------------
numeric_features = train.select_dtypes(include=[np.number])
print( "\n\n{}".format("="*80))
print("Numeric features:\n------------------")
print(numeric_features.dtypes)


# Correlation between features. corr() function uses numeric features only.
# Look at features highest (and most negative) correlated with SalePrice
#-----------------------------------------------------------------------------
c0 = train.corr()
#c0 = numeric_features.corr()   # corr() itself finds the numeric features in data.
print( "\n\n{}".format("="*80))
print("Numeric features highly (or most negatively) correlated with SalePrice:")
print( c0['SalePrice'].sort_values(ascending=False)[:10] )
print( c0['SalePrice'].sort_values(ascending=False)[-5:] )
# Plot of features and their correlation with SalePrice, sorted wrt correlation
plt.figure()
plt.plot(c0.SalePrice.sort_values(),"o-")
plt.xticks(rotation=90)
plt.title("Correlation of features with SalePrice")


# What are the unique values in OverallQuall. 
# Will do this for all features in the uniq() function below.
#-----------------------------------------------------------------------------
print( "\n\n{}".format("="*80))
#print("OverallQual unique values: %s" % train.OverallQual.unique())
print("OverallQual unique values: %s" % sorted(train.OverallQual.unique()))


# Box-plots are another way to look at relationships visually.
# Can do this for other features as well. Seaborn does a nice job again.
#-----------------------------------------------------------------------------
plt.figure()
sns.boxplot(x="Heating",y="logSP",data=train)


# Pairplot of Seaborn is anothre powerful visual tool to investigate relationships.
#-----------------------------------------------------------------------------
plt.figure()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], height = 2.5) # height used to be size in older versions.



# Following are small sample functions to do bits of exploration. 
# They are meant to be edited and ran for exploration purposes, not for production use.

#-----------------------------------------------------------------------------
def uniq():
    '''Print unique values in each of the features. Sort values if possible.'''
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

#-----------------------------------------------------------------------------
def PivT(index='OverallQual', values='SalePrice'):
    '''
    Print a pivot table of the given 2 features, and then plot.
    index is better to be a categorical value with a small cardinality.
    '''
    quality_pivot = train.pivot_table(index=index,
                      values=values, aggfunc=np.median)
    print(quality_pivot)
    quality_pivot.plot(kind='bar', color='blue')
    plt.xlabel(index)
    plt.ylabel('Median {}'.format(values))
    plt.xticks(rotation=0)
  
#-----------------------------------------------------------------------------
def Scat():
    '''Do a scatter plot of 2 features vs. Sale Price'''
    plt.figure()
    plt.scatter(x=train['GrLivArea'], y=target)
    plt.ylabel('Sale Price')
    plt.xlabel('Above grade (ground) living area square feet')

    plt.figure()
    plt.scatter(x=train['GarageArea'], y=target)
    plt.ylabel('Sale Price')
    plt.xlabel('Garage Area')

  