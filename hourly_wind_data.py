import pandas as pd
import os

def process_csv(file):
    df = pd.read_csv(file, usecols=['STATION', 'DATE', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'NAME', 'REPORT_TYPE', 'WND'])

    station_name = df['NAME'][0]

    df[['Angle', 'Angle_Measurement_Quality', 'Wind_Obs_Character', 'Wind_Speed', 'Wind_Speed_Quality']] = df['WND'].str.split(",", expand=True)

    df = df.astype({'Angle': float, 'Angle_Measurement_Quality': float, 'Wind_Obs_Character': str, 'Wind_Speed': float, 'Wind_Speed_Quality': float})

    df = df[(df['Angle'] != 999) & (df['Angle_Measurement_Quality'] == 1) & (df['Wind_Obs_Character'] == 'N') & (df['Wind_Speed'] != 9999) & (df['Wind_Speed_Quality'] == 1) & (df['REPORT_TYPE'] == 'FM-15')]

    df['DATE'] = pd.to_datetime(df['DATE'])
    # df['Year'] = df['DATE'].dt.year
    # df['Month'] = df['DATE'].dt.month
    # df['Day'] = df['DATE'].dt.day
    # df['Hour'] = df['DATE'].dt.hour
    df['Minutes'] = df['DATE'].dt.minute
    df['Seconds'] = df['DATE'].dt.second

    # df = df[['Year', 'Month', 'Day', 'REPORT_TYPE', 'Wind_Speed', 'Angle', 'Hour', 'Minutes', 'Seconds']]
    df = df[['DATE', 'Minutes', 'Seconds', 'REPORT_TYPE', 'Wind_Speed', 'Angle']]
    df = df[(df['Minutes'] == 10) & (df['Seconds'] == 0)]

    output_file = f'Modified_{station_name}.csv'
    df.to_csv(output_file, index=False)
    print(f'Saved modified data to {output_file}')

# Folder containing the CSV files
folder_path = 'D:\\ImportantDocuments\\PhD_MU\\New_Effort\\git_repo_files'

# Iterate over all CSV files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv') and file_name[0].isdigit():
        file_path = os.path.join(folder_path, file_name)
        process_csv(file_path)
