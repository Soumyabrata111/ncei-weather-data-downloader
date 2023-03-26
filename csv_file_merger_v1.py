import os
import pandas as pd

# Set the current working directory
directory = os.getcwd()

# Define the names of the year folders
year_folders = [str(year) for year in range(2012, 2024)]

# Create an empty DataFrame to store the merged data
merged_data = pd.DataFrame()

# Loop through each year folder
for year in year_folders:

    # Construct the path to the year folder
    year_path = os.path.join(directory, year)

    # Loop through each CSV file in the year folder
    for csv_file in os.listdir(year_path):
        if csv_file.endswith(".csv"):

            # Construct the path to the CSV file
            csv_path = os.path.join(year_path, csv_file)

            # Read the data from the CSV file into a DataFrame
            data = pd.read_csv(csv_path, low_memory=False)

            # Append the data to the merged_data DataFrame
            merged_data = pd.concat([merged_data, data])

            # Print a message to indicate that the CSV file has been merged
            print(f"CSV file {csv_file} for year {year} has been merged.")

# Convert all columns to consistent data types
for column in merged_data.columns:
    merged_data[column] = pd.to_numeric(merged_data[column], errors='coerce')

# Group the merged data by station name
grouped_data = merged_data.groupby('STATION')

# Loop through each group and write it to a separate CSV file
for group_name, group_data in grouped_data:
    group_data.to_csv(os.path.join(directory, f"{group_name}.csv"), index=False)
    print(f"{group_name}.csv file has been saved.")
