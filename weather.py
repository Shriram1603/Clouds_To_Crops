from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Load your data (replace with your actual data loading code)
df = pd.read_csv('./Weather_prediction/2022_rain.csv')

data = np.array(df)
lat_lon = data[:, 0:2]
rainfall = data[:, 2:]  # Assuming columns from index 2 onwards are for rainfall
# Get the date range from the column headers
date_strings = df.columns[2:-1]  # Assuming columns from index 2 onwards are the date headers
print(lat_lon)
print(rainfall)
print(date_strings)







