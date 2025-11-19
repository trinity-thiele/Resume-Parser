import pandas as pd
import re # Regular expressions module

# Load dataset
df2 = pd.read_csv('data/raw/resumes_2.csv')

# Print out column names
print(df2.columns.tolist())

# There are two columns, Category and Resume
# Check if any values in each column in the dataframe are null
print(df2.isnull().any())

# Returns false for both columns so we are all good there!

# Print out first couple resumes to view examples
print("RESUME 1")
print(df2.at[0, 'Resume'])
print("RESUME 2")
print(df2.at[1, 'Resume'])
print("RESUME 3")
print(df2.at[2, 'Resume'])

# Rules for cleaning resume string
# - Remove unncessary spaces
# - Remove bullets
# - Remove *
# - Remove unicode characters

# This function should be the same for datasets 2 and 3
def clean_resume(text):
    if pd.isnull(text):
        return ""
    # Remove bullets (• or -) and asterisks
    text = re.sub(r'[•*-]', ' ', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove unicode characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Strip leading/trailing spaces
    return text.strip()

# Apply cleaning function
df2['Resume'] = df2['Resume'].apply(clean_resume)

# Print cleaned resumes
print("CLEANED RESUME 1")
print(df2.at[0, 'Resume'])
print("CLEANED RESUME 2")
print(df2.at[1, 'Resume'])

print(df2.columns.tolist())