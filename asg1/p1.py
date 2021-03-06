#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 10:25:14 2019

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



def bino_likelihood(x_train,y_train,dim):
    p=[0]*dim
    for i in range(dim):
        class_data=x_train[y_train==i+1]
        p[i] = sum(class_data[:,0])/len(class_data)
    return p     
   

def disc(x,prior,mle,dim):
    # p = p(x|c)*p(c)
    p = [mle[i]*prior[i] for i in range(dim)]
    # g = p^x *(1-P)^(1-x)
    g = [p[i] if x==1 else 1-p[i] for i in range(dim)]
    return np.argmax(g)+1

def pred():
    y_pred =np.array([0]*len(x_test))
    
    for i in range(len(x_test)):
        y_pred[i] = disc(x_test[i],prior,mle,dim)
    return y_pred

def confusion_matrix(y_pred,y_test):
    pred= pd.Series(y_pred,name="Predtict")
    act = pd.Series(y_test,name="Actual")
    cf = pd.crosstab(act,pred)
    return cf

data1 = np.array(pd.read_csv("input_1.csv"))
train,test,x_train , y_train ,x_test ,y_test= split_data(data1,0.8,1)
#no.of class  
dim = 2 
#prior   
prior = prior(y_train,dim)
#mle
mle = bino_likelihood(x_train,y_train,dim) 

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


    
