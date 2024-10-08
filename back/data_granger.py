import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from arch import arch_model

data = pd.read_excel('D:/2023semester/Lund University/thesis/Data.xlsx')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

best_lag = {}
granger_causality = {}
max_lag = 5
for column in data.columns:
    for column2 in data.columns:
        if column != column2:
            best_aic = np.inf
            best_lag_val = -1
            for lag in range(1, max_lag + 1):
                test_data = data[[column, column2]]
                result = grangercausalitytests(test_data, maxlag=lag, verbose=False)
                f_value = result[lag][0]['params_ftest'][0]
                p_value = result[lag][0]['params_ftest'][1]
                if p_value < best_aic:
                    best_aic = p_value
                    best_lag_val = lag
            best_lag[(column, column2)] = best_lag_val
            result = grangercausalitytests(data[[column, column2]], maxlag=best_lag_val, verbose=False)
            p_value = result[best_lag_val][0]['ssr_ftest'][1]
            chi_sq_statistic = result[best_lag_val][0]['ssr_ftest'][0]
            granger_causality[(column, column2)] = (best_lag_val, chi_sq_statistic, p_value)

granger_causality = pd.DataFrame(granger_causality.values(), index=granger_causality.keys(), columns=['Best Lag', 'Chi-Sq.Statistic', 'p-value'])
# granger_causality.to_excel('D:/2023semester/Lund University/thesis/granger_causality.xlsx')
# print(granger_causality)















