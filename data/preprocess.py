import pandas as pd
import numpy as np

# Air
air = pd.read_excel('./Air/AirQualityUCI.xlsx', header=0)
air['Date'] = air['Date'].astype('str')
air['Time'] = air['Time'].astype('str')
air['Date'] = air['Date'] + " " + air['Time']
cols = list(air.columns)
cols.remove('Time')
cols.remove('NMHC(GT)')
air = air[cols]
# data -200
cols.remove('Date')
air_values = air[cols].values
mean_list = []
for i in range(air_values.shape[1]):
    values = air_values[:, i]
    mean_list.append(values[values > -200].mean())
mean_list = np.array(mean_list)
mean_list = np.expand_dims(mean_list, axis=0)
mean_list = mean_list.repeat(air_values.shape[0], axis=0)
air_values = np.where(air_values > -200, air_values, mean_list)
df_air = pd.DataFrame(data=air_values, columns=[cols])
df_air.insert(loc=0, column='date', value=air['Date'])
df_air.to_csv('./Air/Air.csv', mode='w', header=True, index=False)


# # River
river = pd.read_csv('./River/RF2.csv')
river = river.iloc[:, :9]
river.rename(columns={"Unnamed: 0": "date"}, inplace=True)
river.to_csv('./River/River.csv', mode='w', header=True, index=False)
