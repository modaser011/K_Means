import numpy as np
import pandas as pd
import collections

data = pd.read_csv('crime_data.csv')
np.set_printoptions(suppress=True)
Values = data.values
firstval= Values[:,0]
Values = Values[:,1:]

numOfCentroid = int(input('Please enter the number of cluster '))

distanceType = input('Please type 1 for Euclidean distance and any another number for Manhattan distance ')

out = int(input('Please enter the outlier threshold '))

def k_means_clustering(Values, numOfCentroid, distanceType):
    centroids = data.sample(n=numOfCentroid).values
    centroids = centroids[:,1:]
    for i in range(len(centroids)):
        print(f'Centroid {i+1}: {centroids[i]}')
    print('-'*100)
    notgood = True
    clusters = {}
    d=0
    while notgood:
        print(f'iteration {d+1}')
        for index, data1 in enumerate(Values):
            list_of_distances = []
            for index1, centroid in enumerate(centroids):
                if distanceType == 1:
                    distance1=0
                    for i in range(len(data1)):
                        distance1+=(centroid[i]-data1[i])**2
                    distance=np.sqrt(distance1) 
                else:
                    distance2=0
                    for i in range(len(data1)):
                        distance2+=abs(centroid[i]-data1[i])
                    distance=abs(distance2) 
                list_of_distances.append(distance)

            clusters[index] = list_of_distances.index(min(list_of_distances))        
        new_centroids = pd.DataFrame(Values).groupby(by=clusters).mean().values
        x=new_centroids
        for i in range(len(x)):
             print(f'New Centroid_{i+1}:{x[i]}')
        print('-'*100)
        if np.count_nonzero((centroids)-(new_centroids)) == 0:
            notgood = False
        else:
            centroids = new_centroids
        d+=1
    return clusters, centroids
clusters, centroids = k_means_clustering(Values, numOfCentroid, distanceType)
distance4 = {}
g=0
for index, rows in enumerate(Values):
    print(f'[{firstval[g]}: {rows}]--> centroid {clusters[index]+1}')
    desired_cluster = clusters[index]
    distance3=0
    for i in range(len(rows)):
        distance3+=(centroids[desired_cluster][i]-rows[i])**2
    dist = np.sqrt(distance3)
    distance4[index, dist] = desired_cluster
    g+=1
print('-'*100)

sort_by_value = dict(sorted(distance4.items(), key=lambda item: item[1]))

new_dictionary = collections.defaultdict(list)
for i, j in sort_by_value.items():
    new_dictionary[j].append(i)

xx=0
d=new_dictionary.items()
for k, val in d :
    print(f'cluter {k+1} has a {len(val)} items')
    sortedItems = sorted(val, key = lambda x: x[1])
    print(f'cluster with distances {k+1} is')
    for i in range(len(sortedItems)):
        print(f'[{firstval[sortedItems[i][0]]}-->{sortedItems[i][1]}]--> cluster {k+1}')
    for i in range(len(sortedItems)):
        if(sortedItems[i][1]>out):
            xx=i
            break
    print('-'*100)
    print(f'outlier of cluster {k+1} :')
    for i in range (xx,len(sortedItems)):
        print(f'outlier of cluter {k+1} -->{firstval[sortedItems[i][0]]}')
    print('-'*100)