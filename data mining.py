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

def k_means_clustering(Values, numOfCentroid, distanceType):
    # Step 1: Initialize random cluster centers (centroids) based on the K
    centroids = data.sample(n=numOfCentroid).values
    centroids = centroids[:,1:]
    for i in range(len(centroids)):
        print(f'Centroid {i+1}: {centroids[i]}')
    print('-'*100)
    # Step 2: Iterate over the data rows, calculate the distance between each row and each one of the random centroids
    # and assign the row to the closest centroid, until convergence which means that the centroids don't change
    isDifference = 1
    clusters = {}
    d=0
    while isDifference:
        print(f'iteration {d+1}')
        for index, data1 in enumerate(Values):
            # Iterating over the rows
            list_of_distances = []
            for cluster_index, centroid in enumerate(centroids):
                # Iterating over the centroids
                if distanceType == 1:
                    # Euclidean distance
                    distance1=0
                    for i in range(len(data1)):
                        distance1+=(centroid[i]-data1[i])**2
                    distance=np.sqrt(distance1) 
                else:
                    distance2=0
                    # Manhattan distance
                    for i in range(len(data1)):
                        distance2+=abs(centroid[i]-data1[i])
                    distance=abs(distance2) 
                list_of_distances.append(distance)

            # Assigning the index to the closest centroid
            clusters[index] = list_of_distances.index(min(list_of_distances))
            # print(clusters)
        
        # Step 3: Calculate the new centroids based on the new cluster assignments
        new_centroids = pd.DataFrame(Values).groupby(by=clusters).mean().values
        x=new_centroids
        for i in range(len(x)):
             print(f'New Centroid_{i+1}:{x[i]}')
        print('-'*100)
        # Step 4: Check if the centroids have changed, if not then we are done
        if np.count_nonzero((centroids)-(new_centroids)) == 0:
            isDifference = 0
        else:
            centroids = new_centroids
        d+=1
    return clusters, centroids


clusters, centroids = k_means_clustering(Values, numOfCentroid, distanceType)

row_to_cluster_distance = {}
g=0


for index, row in enumerate(Values):
    print(f'[{firstval[g]}: {row}]--> centroid {clusters[index]+1}')
    desired_cluster = clusters[index]
    distance3=0
    for i in range(len(row)):
        distance3+=(centroids[desired_cluster][i]-row[i])**2
    dist = np.sqrt(distance3)
    row_to_cluster_distance[index, dist] = desired_cluster
    g+=1

sort_by_value = dict(sorted(row_to_cluster_distance.items(), key=lambda item: item[1]))

