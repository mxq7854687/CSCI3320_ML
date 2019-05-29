from __future__ import print_function
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn import metrics

def create_data():
    # Generate sample points
    centers = [[3,5], [5,1], [8,2], [6,8], [9,7]]
    X, y = make_blobs(n_samples=1000,centers=centers,cluster_std=0.5,random_state=3320)
    # =======================================
    # Complete the code here.
    # Plot the data points in a scatter plot.
    # Use color to represents the clusters.
    # =======================================
    plt.scatter(X[:,0],X[:,1],c=y)
    return [X, y]

def my_clustering(X, y, n_clusters):
    # =======================================
    # Complete the code here.
    # return scores like this: return [score, score, score, score]
    # =======================================
    kmeans=KMeans(n_clusters=n_clusters,random_state=3320).fit(X)
    y_pred=kmeans.labels_
    center = kmeans.cluster_centers_
    
    print("Cluster center is: \n",center)
    plt.scatter(X[:,0],X[:,1],c=y_pred)
    for i in range(len(center)):
        plt.text(center[i][0],center[i][1],str(i+1),color="red",fontsize=12)
    plt.show()
    ari = metrics.cluster.adjusted_rand_score(y,y_pred)
    mri= metrics.cluster.mutual_info_score(y,y_pred)
    v_measure = metrics.cluster.v_measure_score(y,y_pred)
    sc = metrics.cluster.silhouette_score(X,y_pred)
    return [ari,mri,v_measure,sc]

def main():
    X, y = create_data()
    range_n_clusters = [2, 3, 4, 5, 6]
    ari_score = [None] * len(range_n_clusters)
    mri_score = [None] * len(range_n_clusters)
    v_measure_score = [None] * len(range_n_clusters)
    silhouette_avg = [None] * len(range_n_clusters)

    for n_clusters in range_n_clusters:
        i = n_clusters - range_n_clusters[0]
        print("Number of clusters is: ", n_clusters)
        [ari_score[i], mri_score[i], v_measure_score[i], silhouette_avg[i]] = my_clustering(X, y, n_clusters)
        print('The ARI score is: ', ari_score[i])
        print('The MRI score is: ', mri_score[i])
        print('The v-measure score is: ', v_measure_score[i])
        print('The average silhouette score is: ', silhouette_avg[i])

    file = open("description3.txt","w")
    file.write("The best number of clusters is 5.")
    file.close()
    # =======================================
    # Complete the code here.
    # Plot scores of all four evaluation metrics as functions of n_clusters.
    # =======================================

if __name__ == '__main__':
    main()

