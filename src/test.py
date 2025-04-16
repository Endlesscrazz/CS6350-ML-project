import pandas as pd

# Define the path to your CSV file
file_path = 'data/train.csv'

# Load the dataset into a pandas DataFrame
df = pd.read_csv(file_path)

# Display the first 5 rows of the dataframe
print(df.head())