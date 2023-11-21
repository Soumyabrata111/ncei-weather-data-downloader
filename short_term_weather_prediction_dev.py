import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import urllib.request

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

# Change the url to iterate the year from 2016 to 2023 and store the URLs in a list
urls = []
for i in range(2016, 2024):
    urls.append('https://www.ncei.noaa.gov/data/global-hourly/access/' + str(i) + '/')
print(urls)

# Store the URLs in a text file
with open('urls.txt', 'w') as f:
    for url in urls:
        f.write(url + '\n')

# Store the URLs in a CSV file
df = pd.DataFrame(urls)
df.to_csv('urls.csv', index=False, header=False)

# Print the URLs
print(urls)

# Create a folder for each file
for file in files:
    os.makedirs(file[:-4], exist_ok=True)

# Download the files
for file in files:
    for url in urls:
        try:
            urllib.request.urlretrieve(url + file, file[:-4] + '/' + url[-5:-1] + '.csv')
        except Exception as e:
            print(f"Failed to download {url + file}. Error: {e}")

# Print the names of all the CSV files
print(files)

# Print the URLs
print(urls)

# Print the names of all the folders
print(os.listdir())

# Print the names of all the files in each folder
for file in files:
    print(os.listdir(file[:-4]))
