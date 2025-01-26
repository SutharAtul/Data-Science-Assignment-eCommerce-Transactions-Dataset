import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import csv

# Load datasets
customers = pd.read_csv("Customers.csv")
products = pd.read_csv("Products.csv")
transactions = pd.read_csv("Transactions.csv")

# Merge transactions and products on 'ProductID'
merged = pd.merge(transactions, products, on="ProductID", how="left")

# Check for 'Price' column and handle missing values
if 'Price' not in merged.columns:
    print("Warning: 'Price' column is missing. Creating a proxy price from TotalValue and Quantity.")
    merged['Price'] = merged['TotalValue'] / merged['Quantity']  # Calculate proxy price
    merged['Price'] = merged['Price'].fillna(0)  # Fill NaN values with 0


# Merge with customers
merged = pd.merge(merged, customers, on="CustomerID", how="left")

# Create customer profiles
customer_profiles = merged.groupby("CustomerID").agg({
    'TotalValue': 'sum',  # Total transaction value
    'Quantity': 'sum',    # Total quantity purchased
    'Price': 'mean'       # Average product price
}).reset_index()

# Compute similarity matrix using cosine similarity
numerical_features = customer_profiles.drop("CustomerID", axis=1)
similarity = cosine_similarity(numerical_features)

# Generate Lookalike recommendations
recommendations = []
for i, customer in enumerate(customer_profiles['CustomerID']):
    scores = list(enumerate(similarity[i]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:4]  # Top 3 similar customers
    recommendations.append({
        'CustomerID': customer,
        'Lookalikes': [(customer_profiles['CustomerID'][j], round(score, 2)) for j, score in scores]
    })

# Save recommendations to CSV
with open('Lookalike.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['CustomerID', 'Lookalikes'])
    for r in recommendations:
        lookalikes_str = '; '.join([f"({cust_id}, {score:.2f})" for cust_id, score in r['Lookalikes']])
        writer.writerow([r['CustomerID'], lookalikes_str])

print("Lookalike recommendations saved to Lookalike.csv.")
