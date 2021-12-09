import numpy as np

n = int(input("Initial centroids are (0, 0) and (2, 2). Please enter enter the number of additional data points: "))
PointList = [] # Empty list for data points

for i in range(n):
    PointList.append([float(x) for x in input("Please enter a data point separated by a space: ").split()])

dlist = np.array(PointList)

# c1 and c2 are initial centroids
c1 = np.array([0, 0])
c2 = np.array([2, 2])

cluster1 = [] # List for data points in cluster (0, 0)
cluster2 = [] # List for data points in cluster (2, 2)

# Data points are assigned to clusters based on the euclidian distance to centroids
for data_point in PointList:
    cluster1.append(data_point) if np.linalg.norm(c1 - data_point) <= np.linalg.norm(c2 - data_point) else cluster2.append(data_point)


cluster1 = np.array(cluster1)
cluster2 = np.array(cluster2)

# Caluclate new centroids based on data points in clusters
if len(cluster1) != 0:
    c1 = cluster1.mean(axis=0).round(2)
    print(c1)
else:
    print(None)

if len(cluster2) != 0:
    c2 = cluster2.mean(axis=0).round(2)
    print(c2)
else:
    print(None)
