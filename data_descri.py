import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, jarque_bera
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from arch.unitroot import PhillipsPerron


data = pd.read_excel('D:/2023semester/Lund University/thesis/Data.xlsx')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

def compute_statistics(series):
    skewness = skew(series)
    kurt = kurtosis(series)
    jb_stat, jb_p_value = jarque_bera(series)
    adf_stat, adf_p_value, _, _, _, _ = adfuller(series)
    pp_test = PhillipsPerron(series)
    pp_stat, pp_p_value = pp_test.stat, pp_test.pvalue
    kpss_stat, kpss_p_value, _, _ = kpss(series, 'ct', nlags="auto")
    lbq_results = acorr_ljungbox(series.dropna(), lags=[15], return_df=True)
    lbq_stat = lbq_results['lb_stat'].iloc[-1]
    lbq_p_value = lbq_results['lb_pvalue'].iloc[-1]
    lm_stat, lm_p_value, _, _ = het_arch(series.dropna())
    return (
        skewness, kurt, jb_stat, jb_p_value, adf_stat, adf_p_value, pp_stat, pp_p_value,
        kpss_stat, kpss_p_value, lbq_stat, lbq_p_value, lm_stat, lm_p_value
    )

statistics = {}
for column in data.columns:
    mean = data[column].mean()
    std_dev = data[column].std()
    min_val = data[column].min()
    max_val = data[column].max()
    stats = compute_statistics(data[column])
    
    statistics[column] = {
        'Mean': mean,
        'Std. dev.': std_dev,
        'Min.': min_val,
        'Max.': max_val,
        'Skewness': stats[0],
        'Kurtosis': stats[1],
        'JB': stats[2],
        'JB p-value': stats[3],
        'ADF': stats[4],
        'ADF p-value': stats[5],
        'PP': stats[6], 
        'PP p-value': stats[7],
        'KPSS': stats[8], 
        'KPSS p-value': stats[9],
        'LBQ': stats[10],
        'LBQ p-value': stats[11],
        'LM': stats[12],
        'LM p-value': stats[13]
    }

def format_with_significance(stat, p_value):
    significance = ''
    if p_value < 0.01:
        significance = '***'
    elif p_value < 0.05:
        significance = '**'
    elif p_value < 0.1:
        significance = '*'
    return f"{stat:.2f}{significance}"

for column, values in statistics.items():
    for key in ['JB', 'ADF', 'PP', 'KPSS', 'LBQ', 'LM']:
        stat = values[key]
        p_value = values[f'{key} p-value']
        values[key] = format_with_significance(stat, p_value)
        del values[f'{key} p-value'] 

formatted_descriptive_table = pd.DataFrame.from_dict(statistics, orient='index')
formatted_descriptive_table.to_excel('D:/2023semester/Lund University/thesis/descriptive_table.xlsx')

print(formatted_descriptive_table)
