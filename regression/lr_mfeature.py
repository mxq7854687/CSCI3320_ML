import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model,preprocessing
import seaborn
seaborn.set()

names=["symboling","normalized losses","make","fuel-type","aspiration","num-of-doors",
                   "body-style","drive-wheels","engine-location","wheel-base","length","width",
                   "height","curb-weight","engine-type","num-of-cylinders","engine-size","fuel-system",
                   "bore","stroke","compression-ratio","horesepower","peak-rpm","city-mpg","highway-mpg",
                   "price"]
df = pd.read_csv('imports-85.data',
            header=None,
            names = names,
                na_values="?")

#remove missing data
df2 = df.dropna()

#data standardization
x_train,y_train= df2.iloc[:,[23,21,16,22]],df2.iloc[:,25]

def standardization(x_train):
    x_train = np.array(x_train)
    if x_train.ndim ==1:
        x_train = x_train.reshape(-1,1)
    x_scaler = preprocessing.StandardScaler()
    x_train_scaled = x_scaler.fit_transform(x_train)
    return x_train_scaled

x_train_sc = standardization(x_train)
y_train_sc = standardization(y_train)


def ls(x_train_sc,y_train_sc):
    ux = np.insert(x_train_sc,0,1,axis=1)
    tux = np.transpose(ux)
    #step 1 xTx
    #s1 = np.dot(tux,ux)
    #step 2 inverse
    #s2 = np.linalg.inv(s1)
    #step 3 inv xT
    #s3 = np.dot(s2,tux)
    #step 4 theta
    #theta=np.dot(s3,y_train_sc)
    theta = np.dot(np.dot(np.linalg.inv(np.dot(tux,ux)),tux),y_train_sc)
    return theta


#add script print out
print("Parameter theta calculated by normal equation: \n",ls(x_train_sc,y_train_sc))


#Stocastic gradient descent
def get_beta(model):
    beta = np.concatenate((model.intercept_,model.coef_))
    return beta

def sgd(x_train_sc,y_train_sc):
    sgd = linear_model.SGDRegressor(loss="squared_loss")
    y_train_sc= y_train_sc.reshape(len(y_train_sc),)
    sgd.fit(x_train_sc,y_train_sc)
    theta = get_beta(sgd)
    return theta
#add script print out
print("Parameter theta calculated by SGD: \n",tuple(sgd(x_train_sc,y_train_sc)))

