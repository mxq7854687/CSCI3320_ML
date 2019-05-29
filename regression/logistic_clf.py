import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split


n_samples = 10000

centers = [(-1, -1), (1, 1)]
X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.8,
                  centers=centers, shuffle=False, random_state=19)

y[:n_samples // 2] = 0
y[n_samples // 2:] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=19)
log_reg = linear_model.LogisticRegression()

#3.1
log_reg.fit(X_train,y_train)
y_pred = log_reg.predict(X_test)

zero_or_one = (y_pred==0).sum()+(y_pred==1).sum()
zero_or_one==len(y_pred)

X_test1 = X_test[y_pred==1]
X_test0 = X_test[y_pred==0]
plt.scatter(X_test1[:,0],X_test1[:,1],c="r",marker="o",label="class 1")
plt.scatter(X_test0[:,0],X_test0[:,1],c="b",marker="o",label="class 0")
plt.title("Classification with Logistic Regression")
plt.xlabel("feature1")
plt.ylabel("feture2")
plt.legend(loc="upper right")
plt.show()

#3.2
print("Number of wrong predictions is: ",sum(y_pred!=y_test))