# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:19:23 2019

@author: PEKERPCLocal

Description: Explorations on the Kaggle Iowa House Prices competetion.
             Includes functions that test different models on the data.
             Only the numeric features are used in this set of tests.
             Functions usually start with simple tests and trials, and 
             go as a progression through deeper and more focused tests of models,
             reflecting the process of learning. Thus, it is not a production
             library, but a record of exploration.
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import neighbors
from sklearn import neural_network
from sklearn import ensemble
from sklearn import tree
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

import seaborn as sns
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
def corrAnalysis():
    '''
    Initial explorations of the data:
    Inspect correlataion between data columns (input features AND the target).
    DataFrame.corr() function processes numeric columns of the data only.
    '''
    corr_mtrx = train.corr()
    print(corr_mtrx)
    plt.subplots(figsize=(12, 9))
    sns.heatmap(corr_mtrx, vmax=.8, square=True)
    
    hicorr = corr_mtrx.logSP.sort_values(ascending=False)
    #print(hicorr)
    for feat in hicorr.index:
        print("%15s:\t%f" % (feat,hicorr[feat]) )
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
def getNumericXy():
    '''
    Return X and y from the training data, using only the numeric columns.
    Interpolate, and drop NA's.
    y is logSP (logarith of the SalePrice), since that's what the competition uses.
    '''
    data = train.select_dtypes(include=[np.number]).interpolate().dropna()
    y = data.logSP
    X = data.drop(["logSP","SalePrice"],axis=1)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)
    return X,y
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
def tryLinearRegression(X_train, X_test, y_train, y_test):
    '''
    Initial simple exploration on the problem using LinearRegression.
    Test LinearRegression on given train and test (validation) data:
    Print performance metrics R^2 and root mean squared error (RMSE).
    Plot (scatter) predictions vs. actual y_test.
    '''
    lr = linear_model.LinearRegression()
    model = lr.fit(X_train, y_train)
    print ("R^2 is: \n", model.score(X_test, y_test))
    
    predictions = model.predict(X_test)
    print ('RMSE is: \n', np.sqrt(mean_squared_error(y_test, predictions)))
    
    plt.scatter(predictions, y_test, alpha=.7, color='b') #alpha helps to show overlapping data
    plt.xlabel('Predicted Price')
    plt.ylabel('Actual Price')
    plt.title('Linear Regression Model')
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
def loopTrainSize():
    '''
    Explore effect of train/test size on model error.
    Loop thru a range of test size percentages, 
    print and plot how RMSE error changes wrt test size,
    using LinearRegression, and numeric X,y.
    '''
    X, y = getNumericXy()
    r2list = []
    rmselist = []
    for tsize in range(10,80,5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tsize/100.0)
        lr = linear_model.LinearRegression()
        model = lr.fit(X_train, y_train)
        r2 = model.score(X_test, y_test)
        r2list.append(r2)
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        rmselist.append(rmse)
        print("TestSize=%3d\tr2=%.4f\trmse=%.4f" % (tsize,r2,rmse))
        
    plt.plot(range(10,80,5),rmselist)
    plt.title("RMSE vs. test size")
    plt.figure()
    plt.plot(range(10,80,5),r2list)
    plt.title("R^2 vs. test size")
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
def loopTrainSizeCV():
    '''
    Similar to loopTrainSize() function, but uses cross_validation_score, because
    a single error figure turned out to be not reliable.
    
    You can see that the error gets higher as training size gets smaller (test size increases).
    '''
    X, y = getNumericXy()
    r2list = []
    rmselist = []
    for tsize in range(10,80,5):
        lr = linear_model.LinearRegression()
        shuffle_cv = ShuffleSplit(n_splits=100,test_size=tsize/100.0)
        rmse_scores = np.sqrt(-1.0*cross_val_score(lr, X, y, 
                                            cv=shuffle_cv, 
                                            scoring="neg_mean_squared_error"))
        rmselist.append(rmse_scores.mean())
        print("TestSize=%3d\trmse=%.4f" % (tsize,rmse_scores.mean()))
        
    plt.plot(range(10,80,5),rmselist)
    plt.title("RMSE vs. test size")
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
def loopTrainSizeFixed(tsize = 10, rep=300):
    '''
    Run LinearRegression on numeric X,y rep times, with a test size of tsize, and
    plot the distribution of mean squared error.
    Return list of MSE.
    
    Cont. exploring effect of train/test size on model error:
    loopTrainSize() function showed that a single calculation of error 
    does not give a stable/reliable result.
    For this reason, repeat the test rep times, for a given tsize test size percentage.
    Plot the distribution of errors.
    
    Interestingly, distributions are usually multi-modal.
    '''
    X, y = getNumericXy()
    r2list = []
    rmselist = []
    for r in range(rep):
        if r%10==0: print(r,end=",")
        if r%100==0: print()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tsize/100.0)
        lr = linear_model.LinearRegression()
        model = lr.fit(X_train, y_train)
        r2 = model.score(X_test, y_test)
        r2list.append(r2)
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        rmselist.append(rmse)
    #plt.plot(range(rep),rmselist,"o--")
    plt.figure()
    plt.hist(rmselist,max(int(np.sqrt(rep)*1.5),10))
    plt.title("Test size = %d%%. Rep = %d" % (tsize,rep))
    plt.figure()
    sns.distplot(rmselist)
    plt.title("Test size = %d%%. Rep = %d" % (tsize,rep))
    return rmselist
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
def loopTrainSizeFixedMdl(mdl, tsize = 20, rep=300):
    '''
    Run given model rep times, on numeric X,y,
    with a test size percent of tsize, and 
    plot the distribution of mean squared errors.
    Return list of MSE.
    
    Similar to loopTrainSizeFixed(), but model is passed as a parameter.
    '''
    X, y = getNumericXy()
    rmselist = []
    for r in range(rep):
        print(r,end=",")
        if r%30==0:print()
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=tsize/100.0)
        model = mdl.fit(X_train, y_train)
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        rmselist.append(rmse)
    print(model)
    #plt.plot(range(rep),rmselist,"o--")
    plt.figure()
    #plt.hist(rmselist,max(int(np.sqrt(rep)*1.5),10))
    sns.distplot(rmselist)
    plt.title("Test size = %d%%. Rep = %d" % (tsize,rep))
    return rmselist
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
def loopModels(tsize=30,rep=300):
    '''
    Loop through a number of predefined regression modelds and report their performance.
    Use numeric X,y.
    Print mean and median RMSE for each model. 
    Plot the distribution of RMSE for each model.
    Return a dictionary of: {"model name" : RMSE list} that includes all models.
    This dictionary can easily be converted to pd.DataFrame and saved as csv.

    Each model is run rep times, to obtain a distribution of RMSE instead of a single number.
    Usually a single error calculation is done, or cross validation is used (e.g. with 5 fold),
    but a distribution of errors from e.g. 300 runs gives interesting results, useful in exploration.
    '''
    X, y = getNumericXy()
    
    # OverallQual is the feature with highest correlation with the target.
    # Wanted to see how a single, highest correlated feature do by itself.
    # Uncomment if you want to test.
    #X = data[["OverallQual"]]

    models = {
            #"ARDRegression":linear_model.ARDRegression(),
            "BayesianRidge":linear_model.BayesianRidge(),
            "ElasticNet":linear_model.ElasticNet(),
            "HuberRegressor":linear_model.HuberRegressor(),
            "Lars":linear_model.Lars(),
            "Lasso":linear_model.Lasso(),
            "LassoLars":linear_model.LassoLars(),
            "RANSACRegressor":linear_model.RANSACRegressor(),
            "DecisionTree":tree.DecisionTreeRegressor(),
            "RandomForest20":ensemble.RandomForestRegressor(n_estimators=20),
            "RandomForest100":ensemble.RandomForestRegressor(n_estimators=100),
            "RandomForest200":ensemble.RandomForestRegressor(n_estimators=200),
            "XGBRegressor":XGBRegressor(),
            "XGBRegressor_n1000_r05":XGBRegressor(n_estimators=1000, learning_rate=0.05),
            }
    models_rmselist = {}
    for mname,mdl in models.items():
        print("%s\n%s" % ("-"*80,mname))
        rmselist = []
        for r in range(rep):
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=tsize/100.0)
            model = mdl.fit(X_train, y_train)
            predictions = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            rmselist.append(rmse)
        print("\tMean RMSE = %f, Median RMSE = %f" % (np.mean(rmselist), np.median(rmselist)) )
        models_rmselist[mname] = rmselist
        plt.figure()
        plt.hist(rmselist,max(int(np.sqrt(rep)*1.5),10))
        #plt.hist(rmselist,np.linspace(0.01,0.07,num=150))
        plt.title("%s mn=%.4f,md=%.4f (tsize=%d,rep=%d)"%
                  (mname,np.mean(rmselist),np.median(rmselist),tsize,rep))
    return models_rmselist
#-----------------------------------------------------------------------------
        
#-----------------------------------------------------------------------------
def Submission(model, fname = "submission.csv"):
    '''
    Generate Kaggle submission using the given model, and the test data.
    Use only the numeric features.
    Save the csv file, with the given file name.
    '''
    tt = test.select_dtypes(include=[np.number]).interpolate()
    predlog = model.predict(tt)
    pred = np.exp(predlog)
    sub = pd.DataFrame()
    sub["Id"] = test.Id
    sub['SalePrice'] = pred
    sub.to_csv(fname, index=False)
#-----------------------------------------------------------------------------
    
#-----------------------------------------------------------------------------
def KNearestNCA():
    '''
    Test the nearest neighbor regressor.
    Separate function, to play with NCA.
    '''
    nca = neighbors.NeighborhoodComponentsAnalysis(random_state=42)
    knn = neighbors.KNeighborsRegressor()
    nca_pipe = Pipeline([('nca', nca), ('knn', knn)])
    #nca_pipe.fit(X_train, y_train) 
    rmselist = loopTrainSizeFixedMdl(nca_pipe, tsize = 30, rep=1000)
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
def MLPReg(tsize=30,rep=500):
    '''
    Test the MLP (multi-layer perceptron) regressor.
    Use standard scaling of features.
    MLP is said to be sensitive to scaling,
    so this is a separate function for testing MLP with feature scaling.
    '''
    X, y = getNumericXy()
    Xs = StandardScaler().fit_transform(X)
    rmselist = []
    for r in range(rep):
        print(r,end=",")
        if r%30==0:print()
        X_train, X_test, y_train, y_test = train_test_split(Xs,y,test_size=tsize/100.0)
        mdl = neural_network.MLPRegressor(max_iter=1000)
        model = mdl.fit(X_train, y_train)
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        rmselist.append(rmse)
    plt.figure()
    #plt.hist(rmselist,max(int(np.sqrt(rep)*1.5),10))
    sns.distplot(rmselist)
    return rmselist
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
def RandForestReg(tsize=30,rep=500,n=100):
    '''
    Test the nearest neighbor regressor.
    Separate function, to play with StandardScaler, 
    verify that scaling is not important for decision trees.
    '''
    X, y = getNumericXy()
    #Xs = StandardScaler().fit_transform(X)
    rmselist = []
    for r in range(rep):
        print(r,end=",")
        if r%30==0:print()
        #X_train, X_test, y_train, y_test = train_test_split(Xs,y,test_size=tsize/100.0)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=tsize/100.0)
        mdl = ensemble.RandomForestRegressor(n_estimators=n)
        model = mdl.fit(X_train, y_train)
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        rmselist.append(rmse)
    plt.figure()
    #plt.hist(rmselist,max(int(np.sqrt(rep)*1.5),10))
    sns.distplot(rmselist)
    plt.title("RandomForest")
    return rmselist
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
def loopNumericFeatures(tsize=30,rep=100):
    '''
    Test the error rate achieved by each numeric feature by itself.
    Plot the distribution of RMSE.
    Use BayesianRidge, wchich gave the best result with numeric features.
    '''
    X, y = getNumericXy()
    for feat in X.columns:
        rmselist = []
        for r in range(rep):
            print(r,end=",")
            if r%30==0:print(feat)
            X_train, X_test, y_train, y_test = train_test_split(X[[feat]],y,test_size=tsize/100.0)
            mdl = linear_model.BayesianRidge()
            model = mdl.fit(X_train, y_train)
            predictions = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            rmselist.append(rmse)
        plt.figure()
        #plt.hist(rmselist,max(int(np.sqrt(rep)*1.5),10))
        sns.distplot(rmselist)
        plt.title(feat)
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
def loopFeaturesDT(tsize=30,rep=100):
    '''
    *** incomplete attempt playing with categorical features ***
    Test the error rate achieved by each feature (numeric AND categorical) by itself.
    Use decision tree, assuming it would work well with categorical features as well.
    Need to use LabelEncoder to handle string/object features. None's are still a problem.
    '''
    #data = train.select_dtypes(include=[np.number]).interpolate().dropna()
    data = train.interpolate()
    y = data.logSP
    X = data.drop(["logSP","SalePrice"],axis=1)
    Xnum = data.select_dtypes(include=[np.number])
    
    feat1errs = {}
    for feat in X.columns:
        if feat not in Xnum.columns:
            le = LabelEncoder()
            le.fit(X[feat].unique())
            Xf = le.transform(X[feat]).reshape(-1, 1)
        else:
            Xf = X[[feat]]
        rmselist = []
        for r in range(rep):
            X_train, X_test, y_train, y_test = train_test_split(Xf,y,test_size=tsize/100.0)
            #mdl = ensemble.RandomForestRegressor(n_estimators=10)
            mdl = tree.DecisionTreeRegressor()
            model = mdl.fit(X_train, y_train)
            predictions = model.predict(X_test)
            rmse = mean_squared_error(y_test, predictions)
            rmselist.append(rmse)
        print("%14s:\tmean=%f\tmed=%f" % (feat,np.mean(rmselist),np.median(rmselist)))
        feat1errs[feat] = np.median(rmselist)
    return feat1errs
