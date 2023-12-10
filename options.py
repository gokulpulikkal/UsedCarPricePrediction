import pandas as pd

df = pd.read_csv('without_dummies.csv')

manufacturer_options = list(df['manufacturer'].unique())

regions_options = list(df['region'].unique())

vehicle_condition_options = list(df['condition'].unique())

fuel_options = list(df['fuel'].unique())

title_status_options = list(df['title_status'].unique())

transmission_options =list(df['transmission'].unique())

vehicle_sizes =list(df['size'].unique())

colors =list(df['paint_color'].unique())

type = list(df['type'].unique())

size = list(df['size'].unique())

models=list(df['model'].unique())


state=list(df['state'].unique())

drive_options = list(df['drive'].unique())




