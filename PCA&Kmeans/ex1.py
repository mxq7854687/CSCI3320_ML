import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
np.random.seed(3320)

def create_data(x1, x2, x3):
    x4 = -4.0 * x1
    x5 = 10 * x1 + 10
    x6 = -1 * x2 / 2
    x7 = np.multiply(x2, x2)
    x8 = -1 * x3 / 10
    x9 = 2.0 * x3 + 2.0
    X = np.hstack((x1, x2, x3, x4, x5, x6, x7, x8, x9))
    return X

def pca(X):
    '''
    # PCA step by step
    #   1. normalize matrix X
    #   2. compute the covariance matrix of the normalized matrix X
    #   3. do the eigenvalue decomposition on the covariance matrix
    # If you do not remember Eigenvalue Decomposition, please review the linear
    # algebra
    # In this assignment, we use the ``unbiased estimator'' of covariance. You
    # can refer to this website for more information
    # http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.cov.html
    # Actually, Singular Value Decomposition (SVD) is another way to do the
    # PCA, if you are interested, you can google SVD.
    # YOUR CODE HERE!
    '''
    norm_x = X-np.mean(X,axis=0)
    s = np.cov(np.transpose(norm_x))
    V,D = np.linalg.eig(s)
    i = np.argsort(V)[::-1]
    V,D=V[i].reshape(len(D),1),D[i]
    ####################################################################
    # here V is the matrix containing all the eigenvectors, D is the
    # column vector containing all the corresponding eigenvalues.
    return [V, D]

def main():
    N = 1000
    shape = (N, 1)
    x1 = np.random.normal(0, 1, shape) # samples from normal distribution
    x2 = np.random.exponential(10.0, shape) # samples from exponential distribution
    x3 = np.random.uniform(-100, 100, shape) # uniformly sampled data points
    X = create_data(x1, x2, x3)

    ####################################################################
    # Use the definition in the lecture notes,
    #   1. perform PCA on matrix X
    #   2. plot the eigenvalues against the order of eigenvalues,
    #   3. plot POV v.s. the order of eigenvalues
    # YOUR CODE HERE!
    pca1 = pca(X)
    
    print(pca1)
    
    eigval=pca1[0]
    
    order=[i for i in range(1,len(eigval)+1)]
    plt.plot(order,eigval,"o-")
    plt.ylabel("eigen values")
    plt.xlabel("the order of eigen values")
    plt.title("eigenvalues against the order of eigenvalues")
    plt.show()
    
    cumsum = np.cumsum(eigval)/np.cumsum(eigval)[-1]
    plt.plot(order,cumsum,"o-")
    plt.ylabel("POV")
    plt.xlabel("the order of eigen values")
    plt.title("POV against the order of eigenvalues")
    plt.show()
    ####################################################################


if __name__ == '__main__':
    main()

