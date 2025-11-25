import pandas as pd
import re # Regular expressions module

# STEP 1: Load datasets
df1 = pd.read_csv('data/raw/resumes_1.csv')
df2 = pd.read_csv('data/raw/resumes_2.csv')
df3 = pd.read_csv('data/raw/resumes_3.csv')

# STEP 2: Selected desired features
# Select the Category and Text features from dataframe 1
df1 = df1[['Category','Text']]
# Dataframe 2 only has two features: Category & Resume (no selection needed)
# Select the Category and Resume_str features from dataset 3
df3 = df3[['Category', 'Resume_str']]

# In dataset 3, we only want the resumes with category Information Technology
filtered_df3 = df3[df3['Category'] == 'INFORMATION-TECHNOLOGY'].copy()
filtered_df3.reset_index(drop=True, inplace=True)

# STEP 3: Check for null values in both columns of each dataframe
print(df1.isnull().any())
print(df2.isnull().any())
print(filtered_df3.isnull().any())
# False was returned for all columns

# STEP 4: Clean the resume string
# - Remove unnecssary spaces
# - Remove * character and bullets
# - Remove any invalid unicode characters

def clean_resume(text):
    # Remove bullets (• or -) and asterisks
    text = re.sub(r'[•*-]', ' ', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove unicode characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Strip leading/trailing spaces
    return text.strip()


# Apply cleaning function to datasets
# NOTE: In dataframes 1 and 3, this creates an additional column 'Resume'
df1['Resume'] = df1['Text'].apply(clean_resume)
df2['Resume'] = df2['Resume'].apply(clean_resume)
filtered_df3['Resume'] = filtered_df3['Resume_str'].apply(clean_resume)


# STEP 5: Visually verify cleaning process
print("RESUME 1", df1.at[0, 'Resume'])
print("RESUME 2", df2.at[0, 'Resume'])
print("RESUME 3", filtered_df3.at[0, 'Resume'])


# STEP 6: Merge dataframes 1 and 2 into a single dataset
merged_df = pd.concat([df1[['Category','Resume']], df2[['Category','Resume']]], ignore_index=True)
print(merged_df.head())


# STEP 7: Export dataframe to a csv file
try:
    merged_df.to_csv('data/clean/combined_dataset.csv', index=False)
    filtered_df3.to_csv('data/clean/information-tech-set.csv', columns = ['Category', 'Resume'], index=False)
    print("Files saved successfully!")
except Exception as e:
    print("Error saving file:", e)


