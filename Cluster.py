# Import Library
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import preprocessing
from sklearn.cluster  import KMeans
from sklearn.decomposition import PCA

mydata=pd.read_csv('C:\\Users\\Asus\\OneDrive\\เดสก์ท็อป\\Main folder\\My_Project\\Data494_csv.csv')
X=mydata[['Poverty gap at $1.90 a day (2011 PPP) (%)']]

# Select N Cluster
inertia_scores = []
range_n_clusters = range(2, 11)

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    inertia_scores.append(kmeans.inertia_)
plt.figure(facecolor='white')
plt.plot(range_n_clusters, inertia_scores, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia (Sum of Squared Distances)")
plt.title("Elbow Method for Optimal k")
plt.show()


# Create Cluster
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
centers = kmeans.cluster_centers_
plt.figure(facecolor='white')
plt.scatter(X, np.zeros_like(X), c=kmeans.labels_, cmap='viridis', alpha=0.8, label='Data Points')
plt.scatter(centers[:, 0], np.zeros_like(centers), c=['#A0522D', '#556B2F', '#4B4B4B'], marker='x', s=100, label='Centroids')
plt.gca().set_facecolor('white')
plt.xlabel('Poverty gap at $1.90 a day (2011 PPP) (%)')
plt.yticks([])  
plt.title('KMeans Clustering ')
plt.legend()
plt.show()


mydata['Cluster'] = kmeans.labels_
