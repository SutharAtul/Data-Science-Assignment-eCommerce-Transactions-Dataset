import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
customers = pd.read_csv("Customers.csv")
transactions = pd.read_csv("Transactions.csv")

# Merge customers and transactions for clustering
merged = pd.merge(transactions, customers, on="CustomerID")
customer_data = merged.groupby("CustomerID").agg({
    'TotalValue': 'sum',  # Total transaction value
    'Quantity': 'sum',    # Total quantity purchased
}).reset_index()

# Scale data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_data.iloc[:, 1:])

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(scaled_data)

# Evaluate Clustering
db_index = davies_bouldin_score(scaled_data, customer_data['Cluster'])
print(f"Davies-Bouldin Index: {db_index:.2f}")

# Visualize Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=scaled_data[:, 0], y=scaled_data[:, 1],
    hue=customer_data['Cluster'], palette="viridis"
)
plt.title("Customer Clusters")
plt.xlabel("Feature 1 (Scaled)")
plt.ylabel("Feature 2 (Scaled)")
plt.legend(title="Cluster")
plt.show()

# Save cluster assignments
customer_data.to_csv("Customer_Clusters.csv", index=False)
print("Cluster assignments saved to Customer_Clusters.csv.")
