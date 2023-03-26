# Merge all the CSV files in folders 2012 to 2023 into one CSV file assuming that all the folders in the current directory

import os
import glob
import pandas as pd

# Get the current working directory
cwd = os.getcwd()

# Get all the folders in the current directory
folders = glob.glob(cwd + '/*')

# Create a list to store the dataframes
df_list = []

# Loop through all the folders
for folder in folders:
    # Get all the CSV files in the folder
    csv_files = glob.glob(folder + '/*.csv')
    # Loop through all the CSV files
    for csv_file in csv_files:
        # Read the CSV file into a dataframe
        df = pd.read_csv(csv_file, low_memory=False)
        # Append the dataframe to the list
        df_list.append(df)

# Concatenate all the dataframes in the list
df = pd.concat(df_list)

# Write the dataframe to a CSV file
df.to_csv('merged.csv', index=False)

# Group the merged data on STATION column
df_grouped = df.groupby('STATION')

# Save each group as a CSV file
for name, group in df_grouped:
    group.to_csv(str(name) + '.csv', index=False)

