# lms_ass6
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

# Load the dataset
data = pd.read_csv('sales_data.csv')

# Display the first few rows of the dataset
print(data.head())

# 1. Data Cleaning

# Handle missing values
imputer = SimpleImputer(strategy='mean')
data['Quantity'] = imputer.fit_transform(data[['Quantity']])
data['Price'] = imputer.fit_transform(data[['Price']])
data['Revenue'] = imputer.fit_transform(data[['Revenue']])

# Remove duplicates
data = data.drop_duplicates()

# 2. Data Transformation

# Convert 'Date' to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Extract new features from 'Date'
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['DayOfWeek'] = data['Date'].dt.dayofweek

# Encode categorical variables
encoder = OneHotEncoder(sparse=False, drop='first')
encoded_product = encoder.fit_transform(data[['Product']])
encoded_category = encoder.fit_transform(data[['Category']])

# Convert encoded features to DataFrame and merge with original data
encoded_product_df = pd.DataFrame(encoded_product, columns=encoder.get_feature_names_out(['Product']))
encoded_category_df = pd.DataFrame(encoded_category, columns=encoder.get_feature_names_out(['Category']))
data = pd.concat([data, encoded_product_df, encoded_category_df], axis=1)

# Drop original categorical columns
data = data.drop(['Product', 'Category', 'Customer', 'Date'], axis=1)

# Scale numerical features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[['Quantity', 'Price', 'Revenue']])

# Convert scaled features to DataFrame and merge with original data
scaled_features_df = pd.DataFrame(scaled_features, columns=['Quantity_scaled', 'Price_scaled', 'Revenue_scaled'])
data = pd.concat([data.reset_index(drop=True), scaled_features_df.reset_index(drop=True)], axis=1)

# Drop original numerical columns
data = data.drop(['Quantity', 'Price', 'Revenue'], axis=1)

# 3. Feature Engineering

# Create interaction features
data['Quantity_Price'] = data['Quantity_scaled'] * data['Price_scaled']
data['Quantity_Revenue'] = data['Quantity_scaled'] * data['Revenue_scaled']

# 4. Feature Selection

# Select best features using ANOVA F-test
X = data.drop('Revenue_scaled', axis=1)  # Assuming 'Revenue_scaled' is the target variable
y = data['Revenue_scaled']

selector = SelectKBest(score_func=f_classif, k=10)
X_new = selector.fit_transform(X, y)

# 5. Dimensionality Reduction

# Apply PCA
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_new)

# Final dataset ready for modeling
final_data = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])

# Display the first few rows of the final dataset
print(final_data.head())
