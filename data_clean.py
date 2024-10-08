import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, jarque_bera
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt


data = pd.read_excel('D:/2023semester/Lund University/thesis/Commodity.xlsx')
data.set_index('Date', inplace= True)
column_mapping = {
    'GCc1': 'Gold',
    'HGc1': 'Copper',
    'PAc1': 'Palladium',
    'PLc1': 'Platinum',
    'SIc1': 'Silver',
    'CLc1': 'WTI Crude Oil',
    'LCOc1': 'Brent Crude Oil',
    'NGc1': 'Natural Gas',
    'Cc1': 'Corn',
    'CCc1': 'Cocoa',
    'CTc1': 'Cotton',
    'KCc1': 'Coffee',
    'LHc1': 'Lean Hogs',
    'Sc1': 'Soybeans'}

data.rename(columns=column_mapping, inplace=True)
desired_order = ['Gold', 'Silver', 'Palladium', 'Platinum', 'Copper', 'WTI Crude Oil',
                 'Brent Crude Oil', 'Natural Gas', 'Corn', 'Cocoa', 'Cotton', 'Coffee',
                 'Lean Hogs', 'Soybeans']
data = data[desired_order]


log_returns = np.log(data / data.shift(-1))
log_returns = log_returns.dropna()
# log_returns.index = log_returns.index.date
# log_returns.to_excel('D:/2023semester/Lund University/thesis/Data.xlsx')
data = log_returns


## Data processing
# descriptive_table = data.describe()
# print(descriptive_table)

def compute_statistics(series):
    skewness = skew(series)
    kurt = kurtosis(series)
    jb_stat, jb_p_value = jarque_bera(series)
    adf_stat, adf_p_value, _, _, _, _ = adfuller(series)
    pp_stat, pp_p_value, _, _ = kpss(series)
    return skewness, kurt, jb_stat, jb_p_value, adf_stat, adf_p_value, pp_stat, pp_p_value

statistics = {}

for column in data.columns:
    mean = data[column].mean()
    std_dev = data[column].std()
    min_val = data[column].min()
    max_val = data[column].max()
    skewness, kurt, jb_stat, jb_p_value, adf_stat, adf_p_value, pp_stat, pp_p_value = compute_statistics(data[column])
    
    statistics[column] = {
        'Mean': mean,
        'Std. dev.': std_dev,
        'Min.': min_val,
        'Max.': max_val,
        'Skewness': skewness,
        'Kurtosis': kurt,
        'JB': jb_stat,
        'ADF': adf_stat,
        'PP': pp_stat,
        'KPSS': pp_p_value}

descriptive_table = pd.DataFrame.from_dict(statistics, orient='index')
print(descriptive_table)    # table see from https://www.sciencedirect.com/science/article/pii/S0301420723002787#sec3
descriptive_table.to_excel('D:/2023semester/Lund University/thesis/descriptive_table.xlsx')

# Plot all in one picture 
plt.figure(figsize=(12, 8))
for i, column in enumerate(data.columns):
    plt.plot(data.index, data[column], label=column)
plt.xlabel('Date')
plt.ylabel('Return')
plt.title('Log Returns of Commodities')
plt.legend(loc='upper left')
plt.show()

# # Plot one by one 
# for column in data.columns:
#     plt.figure(figsize=(8, 6))
#     plt.plot(data.index, data[column], color='blue')  # You can specify different colors if needed
#     plt.title(f'Log Returns of {column}')
#     plt.xlabel('Date')
#     plt.ylabel('Log Returns')
#     plt.grid(True)
#     plt.show()

# Plot one by one with different colour
colors = plt.cm.tab20(np.linspace(0, 1, len(data.columns)))
for i, column in enumerate(data.columns):
    plt.figure(figsize=(8, 6))
    plt.plot(data.index, data[column], color=colors[i])
    plt.title(f'Log Returns of {column}')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.show()




