import pandas as pd

# Read the data and keep only the relevant columns
df = pd.read_csv('43318099999.csv', usecols=['STATION', 'DATE', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'NAME', 'REPORT_TYPE', 'WND'])

# Store the station name
station_name = df['NAME'][0]

# Split the 'WND' column into relevant columns
df[['Angle', 'Angle_Measurement_Quality', 'Wind_Obs_Character', 'Wind_Speed', 'Wind_Speed_Quality']] = df['WND'].str.split(",", expand=True)

# Convert data types
df = df.astype({'Angle': float, 'Angle_Measurement_Quality': float, 'Wind_Obs_Character': str, 'Wind_Speed': float, 'Wind_Speed_Quality': float})

# Drop rows with faulty data
df = df[(df['Angle'] != 999) & (df['Angle_Measurement_Quality'] == 1) & (df['Wind_Obs_Character'] == 'N') & (df['Wind_Speed'] != 9999) & (df['Wind_Speed_Quality'] == 1) & (df['REPORT_TYPE'] == 'FM-12')]

# Convert the DATE column to datetime format and create new columns for Year, Month, Day, Hour, Minutes, and Seconds
df['DATE'] = pd.to_datetime(df['DATE'])
df[['Year', 'Month', 'Day', 'Hour', 'Minutes', 'Seconds']] = df['DATE'].apply(lambda x: pd.Series([x.year, x.month, x.day, x.hour, x.minute, x.second]))

# Keep only the relevant columns and filter rows with Minutes and Seconds equal to 0
df = df[['Year', 'Month', 'Day', 'REPORT_TYPE', 'Wind_Speed', 'Angle', 'Hour', 'Minutes', 'Seconds']]
df = df[(df['Minutes'] == 0) & (df['Seconds'] == 0)]

# Save the modified dataframe to a new csv file named 'Modified_station_name.csv'
df.to_csv(f'Modified_{station_name}.csv', index=False)
