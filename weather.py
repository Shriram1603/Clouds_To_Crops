from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import pickle




weather = pd.read_csv('rain.csv')

# print(weather.loc['2023-08-01':,:])
# print(weather.apply(pd.isnull).sum()/weather.shape[0])
# print(weather.dtypes)

weather['date']=pd.to_datetime(weather['date'])
weather=weather.iloc[:-3,:]
weather["month_temp"]=weather['temp'].rolling(30).mean()
weather["month_rain"]=weather['rain'].rolling(30).mean()

core_weather=weather.iloc[30:,:].copy()
# Extract year, month, and day from the 'date' column
core_weather['year'] = core_weather['date'].dt.year
core_weather['month'] = core_weather['date'].dt.month
core_weather['day'] = core_weather['date'].dt.day
# Drop the original 'date' column, as it's no longer needed
core_weather = core_weather.drop(columns=['date'])

print(core_weather)
x=core_weather[['latitude', 'longitude', 'year', 'month', 'day']]
y_temp=core_weather[['month_temp']].values.ravel()
y_rain=core_weather[['month_rain']].values.ravel()
print(x)
print(y_temp)
print(y_rain)

#Temperature
x_train,x_test,y_temp_train,y_temp_test=train_test_split(x,y_temp,test_size=0.1,random_state=0)
temp_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
temp_model.fit(x_train, y_temp_train)
y_pred_temp = temp_model.predict(x_test)

#Rain
x_train,x_test,y_rain_train,y_rain_test=train_test_split(x,y_rain,test_size=0.1,random_state=0)
rain_model= GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
rain_model.fit(x_train, y_rain_train)
y_pred_rain = rain_model.predict(x_test)

# Calculate MAE for temperature predictions
mae_temp = mean_absolute_error(y_temp_test, y_pred_temp)
print(f"Mean Absolute Error for Temperature: {mae_temp}")

# Calculate MAE for rainfall predictions
mae_rain = mean_absolute_error(y_rain_test, y_pred_rain)
print(f"Mean Absolute Error for Rainfall: {mae_rain}")


#Pickle file of Temperature
pickle.dump(temp_model,open('temp_model.pkl','wb'))
temp=pickle.load(open('temp_model.pkl','rb'))

#Pickle file of 
pickle.dump(rain_model,open('rain_model.pkl','wb'))
rain=pickle.load(open('rain_model.pkl','rb'))

