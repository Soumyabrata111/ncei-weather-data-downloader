import os
import requests
import pandas as pd
from bs4 import BeautifulSoup

# List of CSV files to download
csv_files = [
    "42809099999.csv",
    "42807099999.csv",
    "43279099999.csv",
    "43278099999.csv",
    "43372099999.csv",
    "43371099999.csv",
    "42647099999.csv",
    "42071099999.csv",
    "42410099999.csv",
    "43194099999.csv",
    "43196099999.csv",
    "43192099999.csv",
    "42027099999.csv",
    "42348099999.csv",
    "43314099999.csv",
    "43333099999.csv",
    "42181099999.csv",
    "42182099999.csv",
    "43057099999.csv",
    "43003099999.csv",
    "43128099999.csv",
    "43128599999.csv",
    "43296099999.csv",
    "43295099999.csv",
    "43302599999.csv",
    "42705699999.csv",
    "43353099999.csv",
    "43360099999.csv",
    "43321099999.csv",
    "43314099999.csv",
    "43315099999.csv",
    "43317099999.csv",
    "43318099999.csv",
    "43352099999.csv",
    "43355099999.csv",
    "43376099999.csv",
    "43301099999.csv",
    "43325099999.csv",
    "42705599999.csv",
    "42754099999.csv",
    "43128099999.csv",
    "42867099999.csv",
    "43198099999.csv",
    "43063099999.csv",
    "43081099999.csv",
    "43086099999.csv",
    "43110099999.csv",
    "43157099999.csv",
    "43213099999.csv",
    "42348099999.csv"
]

# Function to download CSV files for a specific year
def download_csv_files(year):
    url = f'https://www.ncei.noaa.gov/data/global-hourly/access/{year}/'

    # Create a directory for the year if it doesn't exist
    year_folder = str(year)
    if not os.path.exists(year_folder):
        os.makedirs(year_folder)

    for file_name in csv_files:
        file_url = f"{url}{file_name}"
        try:
            with requests.get(file_url, stream=True) as file_request:
                # Check if the request was successful
                if file_request.status_code == 200:
                    file_path = os.path.join(year_folder, file_name)

                    # Save the CSV file to the year folder
                    with open(file_path, 'wb') as file:
                        for chunk in file_request.iter_content(chunk_size=8192):
                            if chunk:
                                file.write(chunk)
                    print(f"Downloaded {file_name} for year {year}.")
                else:
                    print(f"Failed to download {file_name} for year {year}.")
        except Exception as e:
            print(f"Error downloading {file_name} for year {year}: {e}")

# Download data for each year from 2012 to 2023
for year in range(2012, 2024):
    download_csv_files(year)
    print(f"Downloaded files for year {year}.")
