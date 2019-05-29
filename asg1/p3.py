#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 00:33:23 2019

@author: richardlee
"""

import pandas as pd
import numpy as np

data3 = np.array(pd.read_csv("input_3.csv"))

def split_data(data,train_ratio,col):
    trainsize = int(len(data)*train_ratio)
    train = data[0:trainsize]
    test= data[trainsize:]
    x_test,y_test = test[:,0:col],test[:,col]
    x_train , y_train = train[:,0:col],train[:,col]
    return train,test,x_train,y_train,x_test,y_test

def prior(clsarray,classnum):
    clss = np.arange(1,classnum+1)
    pri = [0]*classnum
    for i in range(classnum):
        pri[i] = sum(clsarray==clss[i])/len(clsarray)
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


def confusion_matrix(y_pred,y_test,dim):
    cf = [[0]*dim for i in range(dim)]
    y_test = y_test.astype(int)
    for i ,x  in enumerate(y_pred):
        cf[y_test[i]-1][y_pred[i]-1]+=1
    cf =np.reshape(cf,(dim,dim))
    col_name = [i for i in range(1,dim+1)]
    cf = pd.DataFrame(cf,columns=col_name,index=col_name)
    return cf

data3 = np.array(pd.read_csv("input_3.csv"))
train,test,x_train , y_train ,x_test ,y_test= split_data(data3,0.8,1)
#no.of class  
dim = 4
#prior
prior = prior(y_train,dim)
#mle
mu = mean(x_train,y_train,dim)
var= var(x_train,y_train,dim)

y_pred= pred()
cf = confusion_matrix(y_pred,y_test,dim)

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

