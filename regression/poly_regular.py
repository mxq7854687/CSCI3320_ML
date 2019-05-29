import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

X_train = [[5.3], [7.2], [10.5], [14.7], [18], [20]]
y_train = [[7.5], [9.1], [13.2], [17.5], [19.3], [19.5]]

X_test = [[6], [8], [11], [22]]
y_test = [[8.3], [12.5], [15.4], [19.6]]

def get_beta(model):

    beta = np.concatenate((model.intercept_,model.coef_[0,1:]))
    return tuple(beta)
#2.3.1
poly = PolynomialFeatures(degree=5)
#X_train=np.array(X_train).reshape(len(X_train),1)
#X_test=np.array(X_test).reshape(len(X_test),1)
#y_train=np.array(y_train).reshape(len(y_train),1)
#y_test = np.array(y_test).reshape(len(y_test),1)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

pr_model = LinearRegression()
pr_model.fit(X_train_poly,y_train)
pr_beta = get_beta(pr_model)
pr_score = pr_model.score(X_test_poly,y_test)
print("Linear regression (order 5) score is :",pr_score)

xx = np.linspace(0, 26, 100)
xx_poly = poly.transform(xx.reshape(xx.shape[0], 1))
yy_poly = pr_model.predict(xx_poly)

true ,=plt.plot(X_test,y_test,"ro")
predicted , =plt.plot(xx,yy_poly)
plt.legend([true,predicted ], ["True value", "Predicted value"])
plt.title("Linear regression (order 5) result")
plt.show()

#2.3.2
#Ridge regression
ridge_model = Ridge(alpha=1, normalize=False)
ridge_model.fit(X_train_poly, y_train)
ridge_beta = get_beta(ridge_model)
ridge_score = ridge_model.score(X_test_poly,y_test)
print("Linear regression (order 5) score is :",ridge_score)

xx = np.linspace(0, 26, 100)
xx_poly = poly.transform(xx.reshape(xx.shape[0], 1))
yy_ridge = ridge_model.predict(xx_poly)

true ,=plt.plot(X_test,y_test,"ro")
predicted , =plt.plot(xx,yy_ridge)
plt.legend([true,predicted ], ["True value", "Predicted value"])
plt.title("Ridge regression (order 5) result")
plt.show()

#2.3.3
#Q1

#linear regression (order 1)
lr_model = LinearRegression()
lr_model.fit(X_train,y_train)
lr_score = lr_model.score(X_test,y_test)
score= dict([("lr_model",lr_model),("pr_mmodel",pr_model),("ridge_model",ridge_model)])
max(score)

##Q2
alpha_level =[i for i in range(1,10)]
for i in alpha_level:
    r_model= Ridge(alpha = i,normalize=False)
    r_model.fit(X_train_poly,y_train)
    print("alpha = ",i)
    print(r_model.coef_)
    
##Q3
degree = [i for i in range(2,11)]
p_score = np.zeros(len(degree))
r_score = np.zeros(len(degree))
for i in degree:
    poly = PolynomialFeatures(degree =i)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    p_model = LinearRegression()
    p_model.fit(X_train_poly,y_train)
    p_score[i-2]=p_model.score(X_test_poly,y_test)
    
    r_model= Ridge(alpha = 1,normalize=False)
    r_model.fit(X_train_poly,y_train)
    r_score[i-2]=r_model.score(X_test_poly,y_test)  