from bs4 import BeautifulSoup
import requests
import pandas as pd

# Get links to CSV files
def getCSVLinks(url):
    try:
        html_page = requests.get(url)
    except Exception as e:
        print("An error occurred while making the HTTP request:", e)
        return []
    soup = BeautifulSoup(html_page.content, 'lxml')
    links = []

    for link in soup.findAll('a'):
        url = link.get('href')
        if url[0] != '#' and url.endswith('.csv'):
            links.append(url)

    return links

# Save name of all the csv files in a list
stations = getCSVLinks("https://www.ncei.noaa.gov/data/global-hourly/access/2022/")

# Add base URL to each link to create a full URL
for i in range(len(stations)):
    stations[i] = "https://www.ncei.noaa.gov/data/global-hourly/access/2022/" + stations[i]

# Read and save each CSV file
for i in range(len(stations)):
    try:
        df = pd.read_csv(stations[i], low_memory=False)
    except Exception as e:
        print("An error occurred while reading the CSV file:", e)
        continue
    print(df["STATION"].unique())
    df.to_csv(str(df["STATION"].unique()) + ".csv")
