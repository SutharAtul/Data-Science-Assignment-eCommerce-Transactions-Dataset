import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
customers = pd.read_csv("Customers.csv")
products = pd.read_csv("Products.csv")
transactions = pd.read_csv("Transactions.csv")

# Basic information about datasets
print("Customers Dataset:")
print(customers.info())
print("\nProducts Dataset:")
print(products.info())
print("\nTransactions Dataset:")
print(transactions.info())

# Check for missing values
print("\nMissing Values:")
print("Customers:", customers.isnull().sum())
print("Products:", products.isnull().sum())
print("Transactions:", transactions.isnull().sum())

# Merge datasets for deeper analysis
merged = pd.merge(transactions, customers, on="CustomerID").merge(products, on="ProductID", how="left")

# Visualization Examples
# 1. Total Transactions by Region
region_data = merged.groupby("Region")["TotalValue"].sum().sort_values(ascending=False)
region_data.plot(kind="bar", title="Total Transactions by Region")
plt.ylabel("Total Transaction Value")
plt.show()

# 2. Most Purchased Product Categories
sns.countplot(y="Category", data=merged, order=merged["Category"].value_counts().index)
plt.title("Most Purchased Product Categories")
plt.show()

# 3. Customer Signup by Date
customers["SignupDate"] = pd.to_datetime(customers["SignupDate"])
customers["Year"] = customers["SignupDate"].dt.year
customers.groupby("Year").size().plot(kind="bar", title="Customer Signups by Year")
plt.ylabel("Number of Signups")
plt.show()
