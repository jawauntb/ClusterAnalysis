import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# simple cluster analyzer, no langchain needed lol
file = "/Users/jawaun/langchain_stuff/playing_around/docs/2023sales.csv"

# read the csv data into a pandas dataframe
data = pd.read_csv(file)

# extract the columns that we want to cluster
cluster_data = data[['orders', 'gross_sales', 'discounts', 'returns',
                     'net_sales', 'shipping', 'duties', 'additional_fees', 'taxes', 'total_sales']]

# perform k-means clustering with k=3
kmeans = KMeans(n_clusters=3)
kmeans.fit(cluster_data)

# add the cluster labels to the dataframe
data['cluster'] = kmeans.labels_

# visualize the clusters using a scatter plot
plt.scatter(data['total_sales'], data['net_sales'], c=data['cluster'])
plt.xlabel('Total Sales')
plt.ylabel('Net Sales')
plt.title('Cluster Analysis')
plt.show()
