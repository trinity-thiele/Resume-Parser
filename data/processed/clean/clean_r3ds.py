import pandas as pd

# Load dataset
df3 = pd.read_csv('data/raw/resumes_3.csv')

# Print out column names
print(df3.columns.tolist())

# Select specifically the Category and Resume_str columns
df3 = df3[['Category', 'Resume_str']]

# Check if any values in each column in the dataframe are null
print(df3.isnull().any())

# Returns false for both columns

# Print out first couple resumes to view examples
print("RESUME 1")
print(df3.at[0, 'Resume_str'])
print("RESUME 2")
print(df3.at[1, 'Resume_str'])
print("RESUME 3")
print(df3.at[2, 'Resume_str'])

# Rules for cleaning resume string
# Should be the same no?

