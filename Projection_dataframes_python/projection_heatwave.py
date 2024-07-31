import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dask.dataframe as ddf
from pandas import Series, DataFrame
import warnings
warnings.filterwarnings("ignore")

temp_projection = pd.read_csv('projection_data/TemperatureProjections_us.csv')

temp = ddf.read_csv(r'/global/cfs/cdirs/m1532/Projects_MVP/geospatial/climate_heatwave/heatwave_1960_2020/heatwave_definition2/PRISM_apparent_temp_1960_2019.csv', dtype = {'year': int, 'fips': str, 'state': str}).compute().drop(columns={'Unnamed: 0'})


climate_state = temp.drop('fips', axis = 1)
climate_state = climate_state.groupby(['year', 'month', 'state', 'date']).median()
climate_state = climate_state.reset_index()
climate_state.head()

climate = climate_state.sort_values(['state', 'month', 'year'])
climate['date'] = pd.to_datetime(climate['date'])
df_jul_aug = climate[(climate['date'].dt.month.isin([7, 8])) & (climate['date'].dt.year.between(1981, 2010))]
percentiles = df_jul_aug.groupby('state')['AT_min'].quantile(0.85).reset_index()
percentiles.rename(columns={'AT_min': 'threshold_temp'}, inplace=True)

climate_2019 = climate[climate['year'] == 2019]

pop_projection = ddf.read_csv(r'/pscratch/sd/w/wenhuan/projection_data/pop_projection_state_level.csv', dtype = {'year': int, 'fips': str, 'state': str}).compute().drop(columns={'Unnamed: 0'})

def expand_to_months(row):
    expanded_rows = []
    for month in months:
        expanded_row = {
            'state': row['state'],
            'year': row['year'],
            'month': month,
            'pop': row['pop'],
            'population_percentage': row['population_percentage']
        }
        expanded_rows.append(expanded_row)
    return expanded_rows


months = list(range(1, 13))

expanded_data = []
for _, row in pop_projection.iterrows():
    expanded_data.extend(expand_to_months(row))

pop_projected = pd.DataFrame(expanded_data)


import math

def findApparentTemp(temperature):
    e = 0.61094 * math.exp(17.625 * temperature / (temperature + 243.04))
    A = -1.3 + 0.92 * temperature + 2.2 * e
    return A

def count_heatwave(df):
    df['block'] = (df['is_exceedance'] != df['is_exceedance'].shift(1)).cumsum()
    df_heatwaves = df.groupby(['state', 'block', 'year']).agg(
        start_date=('date_future', 'min'),
        end_date=('date_future', 'max'),
        exceedance_days=('is_exceedance', 'sum')
    )
    df_heatwaves = df_heatwaves.reset_index()
    df_heatwaves = df_heatwaves[df_heatwaves['exceedance_days'] >= 2]
    return df_heatwaves

def split_heatwave(df_heatwave):
    result = []
    for index,row in df_heatwave.iterrows():
        start_date = row['start_date']
        end_date = row['end_date']
        date_range = pd.date_range(start_date, end_date, freq='D')
        state = row['state']
        block = row['block']
        df_date = pd.DataFrame(date_range, columns=['date'])
        df_date['year'] = df_date['date'].dt.year
        df_date['month'] = df_date['date'].dt.month
        df_date['day'] = df_date['date'].dt.day
        df_date['state'] = state
        df_date['block'] = block
        result.append(df_date)
        
    return result



range_list = list(range(30))
data_model = []
for index in range_list:
    model_name = temp_projection.iloc[index].values[0]
    temp_increase = temp_projection.iloc[index].values[1]
    yearly_temp_increase = temp_increase/50
    date_range = pd.date_range(start='2020-01-01', end='2100-12-31', freq='D')
    df_date = pd.DataFrame(date_range, columns=['date'])
    df_date['year'] = df_date['date'].dt.year
    df_date['month'] = df_date['date'].dt.month
    df_date['day'] = df_date['date'].dt.day
    df_date = df_date.merge(climate_2019[['date', 'state', 'tMin']], 
                            left_on=[df_date['date'].dt.month, df_date['date'].dt.day], 
                            right_on=[climate_2019['date'].dt.month, climate_2019['date'].dt.day], 
                            suffixes=('_future', '_2019')).drop(columns=['key_0', 'key_1', 'date_2019'])

    df_date['year_increase'] = df_date['year'] - 2019
    df_date['projected_temperature'] = df_date['tMin'] + df_date['year_increase'] * yearly_temp_increase
    df_date['AT_min_projected'] = df_date['projected_temperature'].apply(findApparentTemp)
    df_climate = df_date.merge(percentiles, on='state', how = 'left')
    df_climate['is_exceedance'] = df_climate['AT_min_projected'] > df_climate['threshold_temp']
    df_climate = df_climate.sort_values(['state', 'month', 'year'])
    df_2040 = df_climate[(df_climate['year'] >= 2020) & (df_climate['year'] < 2040)]
    df_4060 = df_climate[(df_climate['year'] >= 2040) & (df_climate['year'] < 2060)]
    df_6080 = df_climate[(df_climate['year'] >= 2060) & (df_climate['year'] < 2080)]
    df_over80 = df_climate[(df_climate['year'] >= 2080) & (df_climate['year'] <= 2100)]
    heatwave_2040 = count_heatwave(df_2040)
    heatwave_result_2040 = split_heatwave(heatwave_2040)
    heatwave_days_2040 = pd.concat(heatwave_result_2040)
    days_2040 = heatwave_days_2040.drop('date', axis = 1)
    days_2040 = days_2040.groupby(['state', 'year', 'month']).count()
    days_2040 = days_2040.reset_index()
    days_2040 = days_2040.rename(columns = {'day': 'heatwave_days'})
    heatwave_4060 = count_heatwave(df_4060)
    heatwave_result_4060 = split_heatwave(heatwave_4060)
    heatwave_days_4060 = pd.concat(heatwave_result_4060)
    days_4060 = heatwave_days_4060.drop('date', axis = 1)
    days_4060 = days_4060.groupby(['state', 'year', 'month']).count()
    days_4060 = days_4060.reset_index()
    days_4060 = days_4060.rename(columns = {'day': 'heatwave_days'})
    heatwave_6080 = count_heatwave(df_6080)
    heatwave_result_6080 = split_heatwave(heatwave_6080)
    heatwave_days_6080 = pd.concat(heatwave_result_6080)
    days_6080 = heatwave_days_6080.drop('date', axis = 1)
    days_6080 = days_6080.groupby(['state', 'year', 'month']).count()
    days_6080 = days_6080.reset_index()
    days_6080 = days_6080.rename(columns = {'day': 'heatwave_days'})
    heatwave_over80 = count_heatwave(df_over80)
    heatwave_result_over80 = split_heatwave(heatwave_over80)
    heatwave_days_over80 = pd.concat(heatwave_result_over80)
    days_over80 = heatwave_days_over80.drop('date', axis = 1)
    days_over80 = days_over80.groupby(['state', 'year', 'month']).count()
    days_over80 = days_over80.reset_index()
    days_over80 = days_over80.rename(columns = {'day': 'heatwave_days'})
    heatwave_days = [days_2040, days_4060, days_6080, days_over80]
    heatwave_projection = pd.concat(heatwave_days)
    heatwave_projection = heatwave_projection.drop('block', axis = 1)
    projection = heatwave_projection.merge(pop_projected[['state', 'year', 'pop', 'month']], on = ['year','month', 'state'], how = 'right')
    projection['heatwave_days'] = projection['heatwave_days'].fillna(0)
    mean_days = projection['heatwave_days'].mean()
    data_days = {
        'model': [model_name],
        'deltaDays': [mean_days]
    }

    df_hw_projection = pd.DataFrame(data_days)
    data_model.append(df_hw_projection)
    projection.to_csv('projection_data/job_result/heatwave_days_state_level_' + model_name + '.csv')
    

hw_projected = pd.concat(data_model)
hw_projected.to_csv('projection_data/job_result/heatwave_days_delta.csv')
    