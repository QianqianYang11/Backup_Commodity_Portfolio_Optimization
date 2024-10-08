import numpy as np
import pandas as pd
from arch import arch_model


data = pd.read_excel('D:/2023semester/Lund University/thesis/Data.xlsx')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

data_scaled = data * 100

garch_results = {}
params_dict = {}

for commodity in data_scaled.columns:
    try:
        model = arch_model(data_scaled[commodity], vol='Garch', p=1, o=0, q=1, dist='skewt')
        res = model.fit(update_freq=0, disp='off')
        
        garch_results[commodity] = res
        params_dict[commodity] = res.params
    except Exception as e:
        print(f"An error occurred with {commodity}: {e}")

params_df = pd.DataFrame(params_dict)
# params_df.to_excel('D:/2023semester/Lund University/thesis/basic_uni_garch.xlsx')








