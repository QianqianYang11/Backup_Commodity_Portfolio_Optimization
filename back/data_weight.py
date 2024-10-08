import numpy as np
import pandas as pd
import mgarch
import armagarch as ag
import pandas_datareader as web
import matplotlib.pyplot as plt
from RegscorePy import aic, bic
from arch import arch_model
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_excel('D:/2023semester/Lund University/thesis/Data.xlsx')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

equal_weight_weights = pd.read_excel('D:/2023semester/Lund University/thesis/optimization_equal_weight.xlsx', index_col=0)
mean_variance_weights = pd.read_excel('D:/2023semester/Lund University/thesis/optimization_mean_variance.xlsx', index_col=0)
min_variance_weights = pd.read_excel('D:/2023semester/Lund University/thesis/optimization_min_variance.xlsx', index_col=0)
min_correlation_weights = pd.read_excel('D:/2023semester/Lund University/thesis/optimization_min_correlation.xlsx', index_col=0)
sharpe_ratio_weights = pd.read_excel('D:/2023semester/Lund University/thesis/optimization_sharpe_ratio.xlsx', index_col=0)
sortino_ratio_weights = pd.read_excel('D:/2023semester/Lund University/thesis/optimization_sortino_ratio.xlsx', index_col=0)
mean_CVaR_weights = pd.read_excel('D:/2023semester/Lund University/thesis/optimization_mean_CVaR.xlsx', index_col=0)
mean_CVaR_weights = mean_CVaR_weights.reindex(data.index, fill_value=np.nan)

weights = {
    "Equal Weight": equal_weight_weights,
    "Mean-Variance": mean_variance_weights,
    "Min Variance": min_variance_weights,
    "Min Correlation": min_correlation_weights,
    "Sharpe Ratio": sharpe_ratio_weights,
    "Sortino Ratio": sortino_ratio_weights,
    "Mean-CVaR": mean_CVaR_weights   }


portfolio_returns_dict = {}
for weight_name, weight_data in weights.items():
    returns = np.sum(weight_data.values * data.values, axis=1) 
    portfolio_returns_dict[weight_name] = pd.Series(returns, index=data.index)


risk_free_rate = 0.02
significance_level = 0.05

portfolio_stats = {}

for portfolio, returns in portfolio_returns_dict.items():
    average_return = np.mean(returns)
    variance = np.var(returns)
    std_dev = np.sqrt(variance)
    sharpe_ratio = (average_return - risk_free_rate) / std_dev if std_dev != 0 else 0

    sorted_returns = np.sort(returns)
    cutoff_index = int(np.floor(significance_level * len(sorted_returns)))
    cvar = np.mean(sorted_returns[:cutoff_index])

    portfolio_stats[portfolio] = {
        'Average Return': average_return,
        'Variance': variance,
        'Sharpe Ratio': sharpe_ratio,
        'CVaR': cvar         }

stats_df = pd.DataFrame(portfolio_stats).transpose()
output_filepath = 'D:/2023semester/Lund University/thesis/portfolio_performance.xlsx'
stats_df.to_excel(output_filepath)
print('Successful')






