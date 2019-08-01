# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 13:54:20 2019

@author: PEKERPCLocal
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import ensemble
from sklearn import tree
from xgboost import XGBRegressor

from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

#-----------------------------------------------------------------------------
# train and test datasets (full, raw) are loaded and available as globals
#-----------------------------------------------------------------------------
train = pd.read_csv(r'G:\_PROJ0\wk.HousePrices\data\train.csv')
test = pd.read_csv(r'G:\_PROJ0\wk.HousePrices\data\test.csv')
train["logSP"] = np.log(train.SalePrice)


#-----------------------------------------------------------------------------
def missingData(verbose=True):
    '''
    Calculate missing data and its percentage for each feature, sorted from highest.
    Return a pd.DataFrame with this content.
    Also print the top 20 rows.
    code adapted from: https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
    '''
    null_elm_count = train.isnull().sum().sort_values(ascending=False)
    # Another way to get missing count is to use count(), and subtract from full count.
    # I verified that this gives the same results, at least for this dataset.
    #elm_count = train.count()
    #miss_elms = train.shape[0] - elm_count
    percent = null_elm_count/train.shape[0]
    missing_data = pd.concat([null_elm_count, percent, train.dtypes], 
                             axis=1, keys=['Missing', 'Percent', 'DataType'],
                             sort=False)
    if verbose: print(missing_data.head(20))
    return missing_data
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
def handleMissing_v1(limit=1):
    '''
    Version1 of handling missing elements:
    Drop columns that are missing more than limit (=1) elements.
    This drops all columns with missing elements, except Electrical.
    Then drop the ROW where Electrical is missing.
    code from: https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
    '''
    missing_data = missingData(verbose=False)
    cols_drop = missing_data[missing_data.Missing > 1].index
    print("Dropping these columns:")
    print(missing_data.loc[cols_drop,:])
    train_v1 = train.drop(cols_drop, 1)
    train_v1 = train_v1.drop( train_v1.loc[train_v1['Electrical'].isnull()].index )
    #train.Electrical.value_counts()   # alternative: impute with most frequent
    print("Missing data left:%d" % (train_v1.isnull().sum().max()) ) #just checking that there's no missing data missing...
    print("Only these data types are remaining in data: %s" % train_v1.dtypes.unique())
    return train_v1
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
def testModelsCV(X,y,models=None,cv=5):
    '''
    Loop through a number of given modelds and report their performance using 
    cross validation. Use given X,y.
    models is a dictionary as {"model name":sklearn-model-object}
    Print mean and median RMSE for each model. 
    Return a dictionary of: {"model name" : RMSE list} that includes all models.
    This dictionary can easily be converted to pd.DataFrame and saved as csv.
    '''

    if models is None:
        models = {
                #"ARDRegression":linear_model.ARDRegression(),
                "BayesianRidge":linear_model.BayesianRidge(),
                #"ElasticNet":linear_model.ElasticNet(),
                #"HuberRegressor":linear_model.HuberRegressor(),
                #"Lars":linear_model.Lars(),
                #"Lasso":linear_model.Lasso(),
                #"LassoLars":linear_model.LassoLars(),
                #"RANSACRegressor":linear_model.RANSACRegressor(),
                #"DecisionTree":tree.DecisionTreeRegressor(),
                "RandomForest20":ensemble.RandomForestRegressor(n_estimators=50),
                "RandomForest100":ensemble.RandomForestRegressor(n_estimators=150),
                "RandomForest200":ensemble.RandomForestRegressor(n_estimators=300),
                "XGBRegressor":XGBRegressor(),
                "XGBRegressor_n1000_r05":XGBRegressor(n_estimators=1000, learning_rate=0.05),
                }
    models_scores = {}
    for mdl_name, mdl in models.items():
        print("%s\n%s" % ("-"*80,mdl_name))
        rmse_scores = np.sqrt(-1.0*cross_val_score(mdl, X, y, 
                                            cv = cv, 
                                            scoring="neg_mean_squared_error"))
        print("\tMean RMSE = %f, Median RMSE = %f" % (np.mean(rmse_scores), np.median(rmse_scores)) )
        models_scores[mdl_name] = rmse_scores
    return models_scores
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
def getXy(data):
    '''Return X and y, given train data. train should have SalePrice and logSP'''
    y = data.logSP
    X = data.drop(["logSP","SalePrice"],axis=1)
    return X,y
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
def exp1():
    '''Experiment 1: Use hanndleMissing_v1 (drop missing>1). Use pd.get_dummies().'''
    train_v1 = handleMissing_v1()
    train_v2 = pd.get_dummies(train_v1)
    X,y = getXy(train_v2)
    print("Only these data types are remaining in X: %s" % X.dtypes.unique())
    mscores = testModelsCV(X,y)
    df = pd.DataFrame(mscores)
    df1 = pd.concat([df,df.describe()])
    df1.T.to_excel("rmse-scores_CV5_handlemissing-v1.xlsx")
    
def submission1(model=None,filename="submit_XGBR_missingv1_n500_r05.csv"):
    '''
    Submit the test results after exp1(). Test data is prepared according to exp1.
    Made model a parameter so that the function can be used with other models.
    model needs to be already fit.
    '''
    train_v1 = handleMissing_v1()
    train_v2 = pd.get_dummies(train_v1)
    X,y = getXy(train_v2)

    X_test_v1 = test[train_v1.columns.drop(["logSP","SalePrice"])]
    X_test_v2 = pd.get_dummies(X_test_v1)
    # some of the values in train don't appear in test, so some dummy columns are missing
    add_cols = set(X.columns) - set(X_test_v2.columns)
    for col in add_cols:
        X_test_v2[col] = 0

    if model is None:
        model = XGBRegressor(n_estimators=500, learning_rate=0.05)
        #model = XGBRegressor(n_estimators=200, learning_rate=0.1)
        model.fit(X,y)
    predlog = model.predict(X_test_v2[X.columns]) # make sure columns are in the order they are in train
    pred = np.exp(predlog)
    
    sub = pd.DataFrame()
    sub["Id"] = test.Id
    sub['SalePrice'] = pred
    sub.to_csv(filename, index=False)

    
#-----------------------------------------------------------------------------
def exp2():
    '''Experiment 2: search for best RandomForest params.'''
    train_v1 = handleMissing_v1()
    train_v2 = pd.get_dummies(train_v1)
    X,y = getXy(train_v2)
    
    models = {}
    n_est_list = [10,30,100,150,200,250,300,400,500,750,1000]
    for n in n_est_list:
        models["RandomForest_%d"%n] = ensemble.RandomForestRegressor(n_estimators=n)
    
    mscores = testModelsCV(X, y, models=models)
    df = pd.DataFrame(mscores)
    df1 = pd.concat([df,df.describe()])
    df1.T.to_excel("rmse-scores_CV5_handlemissing-v1_RandomForest_n10-1000.xlsx")
    
    # plot the change of error wrt n_estimators
    x = df.describe().T.drop(["count","std"],1)
    x.plot()
    plt.figure()
    plt.plot(n_est_list,df.describe().loc["mean",:])
    plt.title("Mean of CV error")
    plt.figure()
    plt.plot(n_est_list,df.describe().loc["50%",:])
    plt.title("Median of CV error")
    
#-----------------------------------------------------------------------------
def exp3():
    '''Experiment 3: search for best XGBRegressor params.'''
    train_v1 = handleMissing_v1()
    train_v2 = pd.get_dummies(train_v1)
    X,y = getXy(train_v2)
    
    models = {}
    #n_est_list = [10,30,100,150,200,250,300,400,500,750,1000]
    n_est_list = [200,300,400,500,750,1000]
    lrate_list = [0.001,0.01,0.05,0.1,0.25,0.5,1]
    for n in n_est_list:
        for lrate in lrate_list:
            models["XGBR_%d_%f"%(n,lrate)] = XGBRegressor(n_estimators=n,learning_rate=lrate)
    
    mscores = testModelsCV(X, y, models=models)
    df = pd.DataFrame(mscores)
    df1 = pd.concat([df,df.describe()])
    df1.T.to_excel("rmse-scores_CV5_handlemissing-v1_XGBR_n10-1000_lr.001-1.xlsx")
    
    # plot the change of error wrt n_estimators
    # This section became problematic after adding lrate_list to optimization
    x = df.describe().T.drop(["count","std"],1)
    x.plot()
    plt.figure()
    plt.plot(n_est_list,df.describe().loc["mean",:])
    plt.title("Mean of CV error")
    plt.figure()
    plt.plot(n_est_list,df.describe().loc["50%",:])
    plt.title("Median of CV error")
    
    return df1