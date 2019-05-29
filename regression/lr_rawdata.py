import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model,preprocessing
import seaborn
seaborn.set()

#2.1.1
names=["symboling","normalized losses","make","fuel-type","aspiration","num-of-doors",
                   "body-style","drive-wheels","engine-location","wheel-base","length","width",
                   "height","curb-weight","engine-type","num-of-cylinders","engine-size","fuel-system",
                   "bore","stroke","compression-ratio","horesepower","peak-rpm","city-mpg","highway-mpg",
                   "price"]
df = pd.read_csv('imports-85.data',
            header=None,
            names = names,
                na_values="?")
#2.1.2
#remove missing data
df1 = df.dropna()

#2.1.3
#split the data set in 80%train 20%test
def split_data(data,train_ratio):
    trainsize = int(len(data)*train_ratio)
    train = data[0:trainsize]
    test= data[trainsize:]
    return train,test

train, test = split_data(df1,0.8)

#data standardization
x_train,y_train,x_test,y_test = train[names[21]],train[names[25]],test[names[21]],test[names[25]]

def standardization(x_train,x_test):
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    if x_train.ndim ==1:
        x_train = x_train.reshape(-1,1)
    if x_test.ndim ==1 :
        x_test = x_test.reshape(-1,1)
    x_scaler = preprocessing.StandardScaler()
    x_train_scaled = x_scaler.fit_transform(x_train)
    x_test_scaled = x_scaler.transform(x_test)
    return x_train_scaled,x_test_scaled

x_train_sc , x_test_sc = standardization(x_train,x_test)
y_train_sc , y_test_sc = standardization(y_train,y_test)
#2.1.4
#run the regression
def lr_fp(x_train,y_train,x_test):
    lr = linear_model.LinearRegression()
    lr.fit(x_train,y_train)
    y_pred = lr.predict(x_test)
    return y_pred

y_pred = lr_fp(x_train_sc,y_train_sc,x_test_sc)
#plot the data
true ,=plt.plot(x_test_sc,y_test_sc,"ro")
predicted , =plt.plot(x_test_sc,y_pred,"b--")
plt.legend([true,predicted ], ["True value", "Predicted value"])
plt.xlabel("Standardized horsepower")
plt.ylabel("Standardized price")
plt.show()

