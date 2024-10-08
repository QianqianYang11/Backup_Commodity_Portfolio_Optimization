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
epsilon = 1e-6

weights_files = {
    'equal_weight': 'optimization_equal_weight.xlsx',
    'mean_variance': 'optimization_mean_variance.xlsx',
    'min_variance': 'optimization_min_variance.xlsx',
    'min_correlation': 'optimization_min_correlation.xlsx',
    'sharpe_ratio': 'optimization_sharpe_ratio.xlsx',
    'sortino_ratio': 'optimization_sortino_ratio.xlsx',
    'mean_CVaR': 'optimization_mean_CVaR.xlsx'
}

weights = {name: pd.read_excel(f'D:/2023semester/Lund University/thesis/{file}', index_col=0)
           for name, file in weights_files.items()}

mean_CVaR_data = data.loc[:'2023-11-30']

def calculate_performance_over_time(data, weights):
    performance_results = {}
    
    for i in range(len(data)):
        subset_data = data.iloc[:i+1]

        for name, w in weights.items():
            if name == 'mean_CVaR':
                portfolio_data = mean_CVaR_data.loc[:subset_data.index[-1]]  # Adjusted data up to current time
            else:
                portfolio_data = subset_data

            portfolio_returns = portfolio_data.dot(w.T)  
            variance = portfolio_returns.var()

            mean_return = portfolio_returns.mean()
            std_dev = portfolio_returns.std()
            risk_free_rate = 0  
            epsilon = 1e-6 
            sharpe_ratio = (mean_return - risk_free_rate) / (std_dev + epsilon)

            negative_returns = portfolio_returns[portfolio_returns < 0]
            downside_std = negative_returns.std()
            sortino_ratio = mean_return / (downside_std + epsilon)

            alpha = 0.95
            cvar = portfolio_returns.quantile(1 - alpha)

            if i not in performance_results:
                performance_results[i] = {}
            performance_results[i][name] = {
                'Return': mean_return,
                'Variance': variance,
                'Sharpe Ratio': sharpe_ratio,
                'Sortino Ratio': sortino_ratio,
                'CVaR': cvar
            }
    
    return performance_results

results_over_time = calculate_performance_over_time(data, weights)

# for time_point, time_results in results_over_time.items():
#     print(f"Time Point: {time_point}")
#     for portfolio, metrics in time_results.items():
#         print(f"  Portfolio: {portfolio}")
#         for metric, value in metrics.items():
#             print(f"    {metric}: {value}")
#     print()



results_df = pd.DataFrame.from_dict({(i,j): results_over_time[i][j] 
                           for i in results_over_time.keys() 
                           for j in results_over_time[i].keys()},
                       orient='index')

output_excel_file = 'D:/2023semester/Lund University/thesis/performance_results_over_time.xlsx'

results_df.to_excel(output_excel_file)

print('Successful')








# ########################################################################
# portfolio_colors = ['blue', 'yellow', 'green', 'orange', 'darkturquoise','royalblue']
# portfolio_markers = ['.', 'P', 'H', 'X', 'D', 'p']
# plot_size = (15, 8)
# line_width = 2
# x_axis_range = [mean_variance['Date'].min(), mean_variance['Date'].max()]
# y_axis_range = [min(mean_variance['Mean Return'].min(), min_variance['Mean Return'].min(), sharpe_ratio['Mean Return'].min(), sortino_ratio['Mean Return'].min()) - 0.01,
#                 max(mean_variance['Mean Return'].max(), min_variance['Mean Return'].max(), sharpe_ratio['Mean Return'].max(), sortino_ratio['Mean Return'].max()) + 0.01]

# metrics = ['Mean Return','Variance', 'Sharpe Ratio', 'Sortino Ratio', 'VaR']

# def plot_metrics(data, metrics, titles, colors, markers):
#     for metric in metrics:
#         plt.figure(figsize=plot_size)
#         for dataset, title, color, marker in zip(data, titles, colors, markers):
#             if metric in dataset.columns:
#                 plt.plot(dataset['Date'], dataset[metric], label=title, linewidth=line_width, color=color,
#                           marker=marker, markevery=int(len(dataset) / 20))

#         plt.title(f'{metric} Over Time', fontsize=18)
#         plt.xlabel('Date', fontsize=14)
#         plt.ylabel(metric, fontsize=14)
#         plt.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
#         plt.xticks(rotation=45, fontsize=10)
#         plt.yticks(fontsize=10)
#         plt.tight_layout()
#         plt.show()

# datasets = [mean_variance, min_variance, sharpe_ratio, sortino_ratio]
# titles = ['Mean-Variance', 'Minimum Variance', 'Sharpe Ratio', 'Sortino Ratio']
# plot_metrics(datasets, metrics, titles, portfolio_colors, portfolio_markers)
# ###############################################################################



# # Parameters for the plot
# portfolio_colors = ['yellow', 'green', 'orange', 'royalblue']
# portfolio_markers = ['o', 's', 'x', '^']
# line_width = 2
# metrics = ['Mean Return', 'Variance', 'Sharpe Ratio', 'Sortino Ratio', 'VaR']

# fig, axes = plt.subplots(nrows=len(metrics), ncols=1, figsize=(15, 15))  

# for i, metric in enumerate(metrics):
#     ax = axes[i]
#     for dataset, title, color, marker in zip(datasets, titles, portfolio_colors, portfolio_markers):
#         if metric in dataset.columns:
#             ax.plot(dataset['Date'], dataset[metric], label=title, linewidth=line_width, color=color,
#                     marker=marker, markevery=int(len(dataset) / 20))
    
#     ax.set_title(f'{metric} from 1991 to 2023')
#     ax.set_xlabel('Date')
#     ax.set_ylabel(metric)
#     ax.legend(title='Portfolio', bbox_to_anchor=(1.05, 1), loc='upper left')
#     ax.tick_params(axis='x', rotation=45)

# plt.tight_layout()
# plt.show()

# ###########################################################################
# #Sub-period
# import matplotlib.pyplot as plt
# import pandas as pd

# # Your previously defined datasets
# datasets = [mean_variance, min_variance, sharpe_ratio, sortino_ratio]
# titles = ['Mean-Variance', 'Minimum Variance', 'Sharpe Ratio', 'Sortino Ratio']

# # Ensure 'Date' columns are in datetime format for filtering
# for dataset in datasets:
#     dataset['Date'] = pd.to_datetime(dataset['Date'])

# # Define the periods of economic crises
# periods = {    
#     "Stable peroid": ("2014-01-01", "2018-12-31"),
#     "Early 1990s Recession": ("1990-01-01", "1993-12-31"),
#     "Global Financial Crisis": ("2007-08-01", "2009-06-30"),
#     "European Debt Crisis": ("2010-04-01", "2013-12-31"),
#     "COVID-19 Economic Impact": ("2020-01-01", "2022-12-31"),
#     "Russia-Ukraine War": ("2022-02-24", "2023-12-31")
# }

# portfolio_colors = ['yellow', 'green', 'orange', 'royalblue']
# portfolio_markers = ['o', 's', 'x', '^']
# line_width = 2
# metrics = ['Mean Return', 'Variance', 'Sharpe Ratio', 'Sortino Ratio', 'VaR']

# # Function to plot data for specific periods
# def plot_crises_periods(datasets, titles, colors, markers, metrics, periods):
#     for period_name, (start_date, end_date) in periods.items():
#         fig, axes = plt.subplots(nrows=len(metrics), ncols=1, figsize=(15, 15), sharex=True)
#         for i, metric in enumerate(metrics):
#             ax = axes[i]
#             for dataset, title, color, marker in zip(datasets, titles, colors, markers):
#                 period_data = dataset[(dataset['Date'] >= pd.to_datetime(start_date)) & (dataset['Date'] <= pd.to_datetime(end_date))]
#                 if metric in period_data.columns:
#                     ax.plot(period_data['Date'], period_data[metric], label=title, linewidth=line_width, color=color,
#                             marker=marker, markevery=int(len(period_data) / 20))
            
#             ax.set_title(f'{metric} during {period_name}')
#             ax.set_xlabel('Date')
#             ax.set_ylabel(metric)
#             ax.legend(title='Portfolio', bbox_to_anchor=(1.05, 1), loc='upper left')
#             ax.tick_params(axis='x', rotation=45)

#         plt.tight_layout()
#         plt.show()

# plot_crises_periods(datasets, titles, portfolio_colors, portfolio_markers, metrics, periods)

# ##################################################################
# #Table

# import pandas as pd

# # Define the portfolios and their data
# datasets = [mean_variance, min_variance, sharpe_ratio, sortino_ratio]
# titles = ['Mean-Variance', 'Minimum Variance', 'Sharpe Ratio', 'Sortino Ratio']
# metrics = ['Mean Return', 'Variance', 'Sharpe Ratio', 'Sortino Ratio', 'VaR']

# # Define the periods of economic crises
# periods = {    
#     "Stable Period": ("2014-01-01", "2018-12-31"),
#     "Early 1990s Recession": ("1990-01-01", "1993-12-31"),
#     "Global Financial Crisis": ("2007-08-01", "2009-06-30"),
#     "European Debt Crisis": ("2010-04-01", "2013-12-31"),
#     "COVID-19 Economic Impact": ("2020-01-01", "2022-12-31"),
#     "Russia-Ukraine War": ("2022-02-24", "2023-12-31")
# }

# # Ensure 'Date' columns are in datetime format for filtering
# for dataset in datasets:
#     dataset['Date'] = pd.to_datetime(dataset['Date'])

# # Create an empty DataFrame to store results
# results_df = pd.DataFrame()

# # Compute averages for each period
# for period_name, (start_date, end_date) in periods.items():
#     for i, dataset in enumerate(datasets):
#         period_data = dataset[(dataset['Date'] >= pd.to_datetime(start_date)) & (dataset['Date'] <= pd.to_datetime(end_date))]
#         averages = period_data[metrics].mean()
#         averages.name = f"{titles[i]} {period_name}"
#         results_df = pd.concat([results_df, averages], axis=1)

# # Transpose to get the desired table format and rename columns
# results_df = results_df.T
# results_df.columns = metrics

# # Save the results to an Excel file
# output_path = 'D:/2023semester/Lund University/thesis/average_performance.xlsx'
# results_df.to_excel(output_path)

# print("Table of average performance metrics has been saved to:", output_path)
