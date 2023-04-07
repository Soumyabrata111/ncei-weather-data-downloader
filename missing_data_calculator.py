import pandas as pd
import numpy as np

# Read the csv file
# df = pd.read_csv('Modified_CHHATRAPATI SHIVAJI INTERNATIONAL, IN.csv')
# df = pd.read_csv('Modified_CHENNAI INTERNATIONAL, IN.csv')
df = pd.read_csv('Modified_HYDERABAD INTERNATIONAL AIRPORT, IN.csv')

# Create an empty dataframe with columns [Date Year	Month	Day	Hour	Minutes	Seconds]
df1 = pd.DataFrame(columns=['Year', 'Month', 'Day', 'Hour', 'Minutes', 'Seconds'])

# Convert the Date column to datetime format and populate it with hourly timestamps from 2012 to 2022
date_range = pd.date_range(start='2012-01-01', end='2022-12-31', freq='60T')
date_range = date_range + pd.offsets.Minute(10)
df1['Date'] = date_range

# Populate the Year, Month, Day, Hour, Minutes, and Seconds columns of df1
df1['Year'] = df1['Date'].dt.year
df1['Month'] = df1['Date'].dt.month
df1['Day'] = df1['Date'].dt.day
df1['Hour'] = df1['Date'].dt.hour
df1['Minutes'] = df1['Date'].dt.minute
df1['Seconds'] = df1['Date'].dt.second

merged_df = pd.merge(df1, df, on=['Year', 'Month', 'Day', 'Hour', 'Minutes', 'Seconds'], how='outer')
merged_df[['REPORT_TYPE', 'Wind_Speed', 'Angle']] = merged_df[['REPORT_TYPE', 'Wind_Speed', 'Angle']].fillna(value=np.nan)

wind_speed_missing_pct = merged_df['Wind_Speed'].isna().mean() * 100
angle_missing_pct = merged_df['Angle'].isna().mean() * 100

print("Percentage of missing values in Wind_Speed column: {:.2f}%".format(wind_speed_missing_pct))
print("Percentage of missing values in Angle column: {:.2f}%".format(angle_missing_pct))