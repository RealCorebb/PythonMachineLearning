import numpy as np
import random
import matplotlib.pyplot as plt
NUMS=50
def kmeans(X,k):

    # Initialize by choosing k random data points as centroids
    num_features = X.shape[1]
    dataSet = list(X)
    centroids = random.sample(dataSet, k) # Get k centroids
    iterations = 0
    old_labels, labels = [], []

    while not stop_now(old_labels, labels, iterations):
        iterations += 1

        clusters = [[] for i in range(0,k)]
        for i in range(k):
            clusters[i].append(centroids[i])

        # Label points
        old_labels = labels
        labels = []
        for point in X:
            distances = [np.linalg.norm(point-centroid) for centroid in centroids]
            max_centroid = np.argmin(distances)
            labels.append(max_centroid)
            clusters[max_centroid].append(point)

        # Compute new centroids
        centroids = np.empty(shape=(0,num_features))
        for cluster in clusters:
            avgs = sum(cluster)/len(cluster)
            centroids = np.append(centroids, [avgs], axis=0)

    #PLT SHOW#
    for i in range(0,k):
        x=(np.array(clusters[i])[:,0])
        y=(np.array(clusters[i])[:,1])
        plt.scatter(x,y)
        plt.scatter(centroids[i,0],centroids[i,1],marker='X',s=200)
    plt.show()
    return labels
    
def stop_now(old_labels, labels, iterations):
    count = 0
    if len(old_labels) == 0:
        return False
    for i in range(len(labels)):
        count += (old_labels[i] != labels[i])
   # print(count)
    if old_labels == labels or iterations == 2000:
        return True
    return False

if __name__ == '__main__':
    point1 = np.random.normal(loc=[1, 1], scale=[0.5, 0.5] ,size=(NUMS, 2))
    point2 = np.random.normal(loc=[3, 3], scale=[0.5, 0.5] ,size=(NUMS, 2))
    point3 = np.random.normal(loc=[1, 3], scale=[0.5, 0.5] ,size=(NUMS, 2))   
    point4 = np.random.normal(loc=[3, 1], scale=[0.5, 0.5] ,size=(NUMS, 2))
    dataSet = np.concatenate((point1,point2,point3,point4))
    kmeans (dataSet,4)
