#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:23:06 2019

@author: richardlee
"""


import pandas as pd
import numpy as np



def split_data(data,train_ratio,col):
    trainsize = int(len(data)*train_ratio)
    train = data[0:trainsize]
    test= data[trainsize:]
    x_test,y_test = test[:,0:col],test[:,col]
    x_train , y_train = train[:,0:col],train[:,col]
    return train,test,x_train,y_train,x_test,y_test

def prior(clsarray,classnum):
    pri = [sum(clsarray==i+1)/len(clsarray) for i in range(2)]
    return pri

def mean(x_train,y_train,dim):
    mu = [0]*dim
    for i in range(dim):
        class_data=x_train[y_train==i+1]
        mu[i] = np.mean(class_data)
    return mu

def var(x_train,y_train,dim):
    var = [0]*dim
    for i in range(dim):
        class_data=x_train[y_train==i+1]
        var[i] = np.var(class_data)
    return var

def n_pdf(x,mu,var):
    npdf = -(x-mu)**2/var/2
    return npdf

def disc(x,prior,dim):
    # g = -log(var)/2 + normal_density +log(prior)
    g = [-np.log(var[i])/2 +n_pdf(x,mu[i],var[i])+np.log(prior[i]) for i in range(dim)]
    return np.argmax(g)+1  
def pred():
    y_pred =np.array([0]*len(x_test))
    for i in range(len(x_test)):
        y_pred[i] = disc(x_test[i],prior,dim)
    return y_pred

def confusion_matrix(y_pred,y_test):
    y_test = y_test.astype(int)
    pred= pd.Series(y_pred,name="Predtict")
    act = pd.Series(y_test,name="Actual")
    cf = pd.crosstab(act,pred)
    return cf

data2 = np.array(pd.read_csv("input_2.csv"))
train,test,x_train , y_train ,x_test ,y_test= split_data(data2,0.8,1)
#no.of class  
dim = 2
#prior
prior = prior(y_train,dim)
#mle
mu = mean(x_train,y_train,dim)
var= var(x_train,y_train,dim)

y_pred=pred()
cf = confusion_matrix(y_pred,y_test)

accuracy = sum(np.diag(cf))/sum(np.sum(cf))

#get diagonal element(true positive)  
diag = np.diag(cf)

precision =  [diag[i]/sum(cf[i+1])for i in range(dim)]

recall = [diag[i]/np.sum(cf,axis=1)[i+1] for i in range(dim)]

f1 = [2*precision[i]*recall[i]/(precision[i]+recall[i])for i in range(dim)] 
 
ave_f1 = np.mean(f1)
     
def print_summary():
    print("confusion matrix : \n",cf)
    print("accuracy : " ,accuracy)
    print("precision : ",precision)
    print("recall : ",recall)
    print("f1 score : ",f1)
    print("average f1 score : ",ave_f1)

print_summary()