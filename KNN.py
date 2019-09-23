import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
def KNNclassify(inputpoint,dataSet,lables,k):
    m,n = dataSet.shape
    distances=[]
    for i in range(m):
        sum=0
        for j in range(n):
            sum+=(inputpoint[j]-dataSet[i][j])**2
        distances.append(sum**0.5)

    sortedDistance=sorted(distances)

    classCount={}
    for i in range(k):
        voteLabel = labels[ distances.index(sortedDistance[i])]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sortedClass = sorted(classCount.items(), key=lambda d:d[1], reverse=True)
    return sortedClass[0][0]


if __name__ == '__main__':
    data = make_blobs(n_samples=400,centers=2,random_state=8)
    group_org, labels_org = data
    group,group_test,labels,labels_test = train_test_split(group_org,labels_org,test_size=0.3,random_state=0)
    colors = ["magenta","yellow"]
    
    plt.scatter(group[:,0],group[:,1],c=labels,cmap=plt.cm.spring,edgecolor='k')
    x,y = group_test.shape

    for i in range(x):
        r = KNNclassify([group_test[i][0], group_test[i][1]], group, labels, 3)
        print("类别为:")
        print(r)
        color = colors[r]
        plt.scatter(group_test[i][0],group_test[i][1],c=color,marker='*',cmap=plt.cm.spring,edgecolor='k')
    plt.show()

