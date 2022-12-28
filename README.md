# ncei-weather-data-downloader
A Python script that downloads and processes global hourly weather data from the NOAA National Centers for Environmental Information (NCEI) website. Uses Beautiful Soup and pandas. Saves processed data to files named after the weather station.

## Dependencies
This script requires the following dependencies, which are listed in the __requirements.txt__ 

These dependencies can be installed using pip, the Python package manager, by running the following command:<br>
__pip install -r requirements.txt__

## Usage
To use this script, clone this repository and run the script using Python:<br>
__python ncei-weather-data-downloader.py__
<br> The script will scrape links to CSV files from the [NCEI website](https://www.ncei.noaa.gov/data/global-hourly/access/2022/), and then download and process each CSV file. It will extract the unique values in the STATION column, which represents the name of the weather station, and save each CSV file to a new file with the name of the weather station as the filename.

## License
This repository is licensed under the MIT License. This means that others are free to use, distribute, and modify this work as long as they include a copy of the license and attribute the original creators
