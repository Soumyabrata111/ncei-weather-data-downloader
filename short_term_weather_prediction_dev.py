# Store the names of all the CSV files in the URL https://www.ncei.noaa.gov/data/global-hourly/access/2023/ in a list

import requests
from bs4 import BeautifulSoup
import pandas as pd

# Get the HTML content of the URL
url = 'https://www.ncei.noaa.gov/data/global-hourly/access/2023/'
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')

# Get the names of all the CSV files in the URL
files = []
for link in soup.find_all('a'):
    if link.get('href').endswith('.csv'):
        files.append(link.get('href'))

# Store the names of all the CSV files in a text file
with open('files.txt', 'w') as f:
    for file in files:
        f.write(file + '\n')

# Store the names of all the CSV files in a CSV file
df = pd.DataFrame(files)
df.to_csv('files.csv', index=False, header=False)

# Print the names of all the CSV files 
print(files)

