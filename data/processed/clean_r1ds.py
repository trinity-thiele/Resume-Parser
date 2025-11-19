import pandas as pd

# Load dataset
df1 = pd.read_csv('data/raw/resumes_1.csv')

# Print out column names
print(df1.columns.tolist())

# Select the following features 
# Category, Summary, Skills, Experience, Education, Text
df1 = df1[['Category', 'Summary','Skills','Education','Text']]

# Check if any values in each column in the dataframe are null
print(df1.isnull().any())

# Returns false for all columns

# This is already the features we need and want
# Print out first couple row values but I don't believe there is anything to "clean" here
print(df1.head())

# Only debate would be if we need the summary or plain text categories at all
# Take a look at the first summary and plain text
print("SUMMARY 1")
print(df1.at[0, 'Summary'])
print("PLAIN TEXT 1")
print(df1.at[1, 'Text'])

