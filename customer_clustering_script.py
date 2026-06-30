import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = {
    'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Annual_Income': [15, 16, 17, 18, 19, 50, 55, 60, 65, 70],
    'Spending_Score': [39, 81, 6, 77, 40, 50, 55, 52, 60, 65]
}


df = pd.DataFrame(data)

# 2. Select features for clustering
X = df[['Annual_Income', 'Spending_Score']]


kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)


print("Customer Segmentation Results")
print("-----------------------------")
print(df.to_string(index=False))


plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    df['Annual_Income'],
    df['Spending_Score'],
    c=df['Cluster'],
    cmap='viridis',
    s=100,
    edgecolor='k',
    alpha=0.8,
    label='Customers'
)

# Plot cluster centroids
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    s=250,
    c='red',
    marker='X',
    edgecolor='black',
    label='Centroids'
)
plt.title('Customer Segments using K-Means Clustering', fontsize=14, pad=15)
plt.xlabel('Annual Income (k$)', fontsize=12)
plt.ylabel('Spending Score (1-100)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='best')
plt.tight_layout()
plt.show()


