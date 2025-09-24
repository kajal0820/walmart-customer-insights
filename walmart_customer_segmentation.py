import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


df = pd.read_csv('Walmart_customer_purchases.csv')
print(df.head())
df = df.drop(['Customer_ID'], axis=1)
df = pd.get_dummies(df, columns=['Gender'], drop_first=True)
print(df.isnull().sum())
customer_df = df.groupby('Customer_ID').agg(
    Total_Spend=('Purchase_Amount', 'sum'),
    Avg_Spend=('Purchase_Amount', 'mean'),
    Visit_Count=('Purchase_ID', 'count'), # Assuming a unique Purchase_ID
    Discount_Count=('Discount_Applied', 'sum'),
    Avg_Rating=('Rating', 'mean')
).reset_index()