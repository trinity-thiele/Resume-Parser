import pandas as pd

# Load datasets
resumes_1 = pd.read_csv('data/raw/resumes_1.csv')
resumes_2 = pd.read_csv('data/raw/resumes_2.csv')
resumes_3 = pd.read_csv('data/raw/resumes_3.csv')

# Examine datasets
num_rows_resumes_1 = len(resumes_1)
num_rows_resumes_2 = len(resumes_2)
num_rows_resumes_3 = len(resumes_3)
num_unique_categories_resumes_1 = resumes_1['Category'].nunique()
num_unique_categories_resumes_2 = resumes_2['Category'].nunique()
num_unique_categories_resumes_3 = resumes_3['Category'].nunique()
unique_categories_resumes_1 = resumes_1['Category'].unique()
unique_categories_resumes_2 = resumes_2['Category'].unique()
unique_categories_resumes_3 = resumes_3['Category'].unique()
columns_d3 = resumes_3.columns

# Find common categories between two datasets
# First normalize case
unique_categories_resumes_1 = [c.upper() for c in unique_categories_resumes_1]
unique_categories_resumes_2 = [c.upper() for c in unique_categories_resumes_2]
common_categories = set(unique_categories_resumes_1).intersection(set(unique_categories_resumes_2))

# Print observations
print("Number of rows in resumes_1.csv:", num_rows_resumes_1)
print("Number of unique categories in resumes_1.csv:", num_unique_categories_resumes_1)
print("Unique categories in resumes_1.csv:", unique_categories_resumes_1)
print("=" * 50)
print("Number of rows in resumes_2.csv:", num_rows_resumes_2)
print("Number of unique categories in resumes_2.csv:", num_unique_categories_resumes_2)
print("Unique categories in resumes_2.csv:", unique_categories_resumes_2)
print("=" * 50)
print("Common categories:", common_categories)
print("=" * 50)
print("Additional tech-focused dataset:")
print("Number of rows in resumes_3.csv:", num_rows_resumes_3)
print("Number of unique categories in resumes_3.csv:", num_unique_categories_resumes_3)
print("Unique categories in resumes_3.csv:", unique_categories_resumes_3)
print("Columns in resumes_3.csv:", columns_d3)