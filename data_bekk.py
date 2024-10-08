import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, jarque_bera
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from arch.unitroot import PhillipsPerron
import mgarch
import statsmodels.api as sm
from arch import arch_model
from scipy.optimize import fmin, minimize
from scipy.stats import t
from math import inf
from IPython.display import display
import bs4 as bs
import requests
from scipy.stats import t
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from sstudentt import SST
from scipy.optimize import minimize
import cvxpy as cp

data = pd.read_excel('D:/2023semester/Lund University/thesis/Data.xlsx')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
T, N = data.shape
print(data.columns)
print(data.shape)

#covariance
qt_data = pd.read_csv('D:/2023semester/Lund University/thesis/ccc_t_cov.csv')
matrices = []
for index, row in qt_data.iterrows():
    matrix = np.zeros((3, 3))
    for i in range(1, 4):
        for j in range(i, 4):
            if i == j:
                var_col = f'Var(Equation({i}))'
                matrix[i-1, j-1] = row[var_col]
            else:
                cov_col = f'Cov(Equation({i}),Equation({j}))'
                value = row[cov_col]
                matrix[i-1, j-1] = value
                matrix[j-1, i-1] = value
    matrices.append(matrix)
Qt = np.array(matrices)
Qt = np.transpose(Qt, axes=(1, 2, 0))
print("Qt:", Qt.shape)
#correlation
rt_data = pd.read_csv('D:/2023semester/Lund University/thesis/ccc_t_cor.csv')
correlation_matrices = []
for index, row in rt_data.iterrows():
    matrix = np.zeros((3, 3))
    for i in range(1, 4):  
        for j in range(i, 4):
            if i == j:
                matrix[i-1, j-1] = 1
            else:
                cor_col = f'Cor(Equation({i}),Equation({j}))'
                value = row[cor_col] if cor_col in row else 0  
                matrix[i-1, j-1] = value
                matrix[j-1, i-1] = value  
                
    correlation_matrices.append(matrix)
Rt = np.array(correlation_matrices)
Rt = np.transpose(Rt, axes=(1, 2, 0))
print("Rt", Rt.shape)
#mean
mu_values = data.mean()
################################################mean-variance 
######################################################################
#mean-variance 
dates = data.index
T, n_assets = data.shape
gamma = 1
mu = mu_values.values
w = cp.Variable(n_assets)

returns = mu.T @ w
risk = cp.quad_form(w, Qt[:,:,0]) 
objective = cp.Maximize(returns - gamma/2 * risk)
constraints = [cp.sum(w) == 1, w >= 0]
problem = cp.Problem(objective, constraints)
problem.solve()
optimal_weights = w.value

#Time-varying
optimal_weights_time_varying = np.zeros((n_assets, T))
for t in range(T):
    Q_t = Qt[:,:,t]
    risk = cp.quad_form(w, Q_t)
    objective = cp.Maximize(returns - gamma/2 * risk)
    problem = cp.Problem(objective, constraints)
    problem.solve()
    if problem.status not in ["infeasible", "unbounded"]:
        optimal_weights_time_varying[:, t] = w.value
    else:
        print(f"Optimization issue at time {t}: {problem.status}")
        optimal_weights_time_varying[:, t] = np.nan  # Use nan to indicate failed optimization

commodity_names = ['Metal', 'Energy', 'Agricultural']
mean_variance_weight = optimal_weights_time_varying.T
mean_variance_weight = pd.DataFrame(mean_variance_weight, index=dates, columns=commodity_names)
output_excel_path = 'D:/2023semester/Lund University/thesis/optimization_mean_variance.xlsx'
mean_variance_weight.to_excel(output_excel_path)
print("Successful:Mean Variance")
###########################################################################################

# ####################################################################mimimize risk
w = cp.Variable(n_assets)

risk = cp.quad_form(w, Qt[:, :, 0])
objective = cp.Minimize(risk)
constraints = [cp.sum(w) == 1, w >= 0]
problem = cp.Problem(objective, constraints)
problem.solve()
optimal_weights_min_variance = w.value

#Time-varying
optimal_weights_time_varying_min_variance = np.zeros((n_assets, T))

for t in range(T):
    Q_t = Qt[:, :, t]
    risk = cp.quad_form(w, Q_t)
    objective = cp.Minimize(risk)
    problem = cp.Problem(objective, constraints)
    problem.solve()
    if problem.status not in ["infeasible", "unbounded"]:
        optimal_weights_time_varying_min_variance[:, t] = w.value
    else:
        print(f"Optimization issue at time {t}: {problem.status}")
        optimal_weights_time_varying_min_variance[:, t] = np.nan

optimal_weights_min_variance_transposed = optimal_weights_time_varying_min_variance.T


min_variance_weight = pd.DataFrame(optimal_weights_min_variance_transposed, index=dates, columns=commodity_names)
output_excel_path = 'D:/2023semester/Lund University/thesis/optimization_min_variance.xlsx'
min_variance_weight.to_excel(output_excel_path)
print("Successful: Min-Variance")
######################################################################################
######################################################################################
##Sharpe ratio
Rf = 0.0002

def objective(weights, mu, Q_t, Rf):
    portfolio_return = np.dot(weights, mu)
    portfolio_variance = np.dot(weights.T, np.dot(Q_t, weights))
    sharpe_ratio = (portfolio_return - Rf) / np.sqrt(portfolio_variance)
    return -sharpe_ratio 

constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  
bounds = tuple((0, 1) for _ in range(n_assets))
initial_weights = np.ones(n_assets) / n_assets 

optimized_weights = np.zeros((T, n_assets))

for t in range(T):
    Q_t = Qt[:, :, t] 
    result = minimize(objective, initial_weights, args=(mu, Q_t, Rf), method='SLSQP', bounds=bounds, constraints=constraints)
    if result.success:
        optimized_weights[t] = result.x
    else:
        print(f"Optimization issue at time {t}: {result.message}")
        optimized_weights[t] = np.nan  

Sharpe_Ratio_weight = pd.DataFrame(optimized_weights, index=dates, columns=commodity_names)
output_excel_path = 'D:/2023semester/Lund University/thesis/optimization_sharpe_ratio.xlsx'
Sharpe_Ratio_weight.to_excel(output_excel_path)
print("Successful: Sharpe Ratio")

###########################################################################Sortino ratio
###########################################################################Sortino ratio
#Sortino ratio
returns_matrix = data.values

def calculate_downside_risk(weights, returns, Rf):
    downside_returns = np.minimum(returns - Rf, 0)
    downside_variance = np.dot(weights.T, np.dot(np.cov(downside_returns, rowvar=False), weights))
    return np.sqrt(downside_variance)

def objective_sortino(weights, expected_returns, returns, Rf):
    portfolio_return = np.dot(weights, expected_returns)
    downside_risk = calculate_downside_risk(weights, returns, Rf)
    sortino_ratio = (portfolio_return - Rf) / downside_risk
    return -sortino_ratio

constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
bounds = tuple((0, 1) for _ in range(n_assets))
initial_weights = np.ones(n_assets) / n_assets

optimized_weights_sortino = np.zeros((T, n_assets))

for t in range(T):
    Return_t = returns_matrix[t, :]  
    result = minimize(objective_sortino, initial_weights, args=(mu, Return_t, Rf), method='SLSQP', bounds=bounds, constraints=constraints)
    if result.success:
        optimized_weights_sortino[t, :] = result.x
    else:
        print(f"Optimization issue at time {t}: {result.message}")
        optimized_weights_sortino[t, :] = np.nan

Sortino_Ratio_weight = pd.DataFrame(optimized_weights_sortino, index=dates, columns=commodity_names)
output_excel_path = 'D:/2023semester/Lund University/thesis/optimization_sortino_ratio.xlsx'
Sortino_Ratio_weight.to_excel(output_excel_path)
print("Successful: Sortino Ratio")



############################################################################################################
######################################################################################
#minimum correlation
w = cp.Variable(n_assets)

correlation = cp.quad_form(w, Rt[:, :, 0])
objective = cp.Minimize(correlation)
constraints = [cp.sum(w) == 1, w >= 0]
problem = cp.Problem(objective, constraints)
problem.solve()
optimal_weights_min_correlation = w.value

#Time-varying
optimal_weights_time_varying_min_correlation = np.zeros((n_assets, T))

for t in range(T):
    R_t = Rt[:, :, t]
    correlation = cp.quad_form(w, R_t)
    objective = cp.Minimize(correlation)
    problem = cp.Problem(objective, constraints)
    problem.solve()
    if problem.status not in ["infeasible", "unbounded"]:
        optimal_weights_time_varying_min_correlation[:, t] = w.value
    else:
        print(f"Optimization issue at time {t}: {problem.status}")
        optimal_weights_time_varying_min_correlation[:, t] = np.nan

optimal_weights_min_correlation_transposed = optimal_weights_time_varying_min_correlation.T



min_correlation_weight = pd.DataFrame(optimal_weights_min_correlation_transposed, index=dates, columns=commodity_names)
output_excel_path = 'D:/2023semester/Lund University/thesis/optimization_min_correlation.xlsx'
min_correlation_weight.to_excel(output_excel_path)
print("Successful: Min-correlation")

#############################################################################################################
#######################################################################################
#mean-CVaR
alpha = 0.95
window_size = 21
n_assets = data.shape[1]
n_days = data.shape[0]

optimal_weights_mean_CVaR = np.zeros((n_days - window_size + 1, n_assets))

for i in range(n_days - window_size + 1):
    historical_window = data[i:i + window_size].values
    sorted_returns = np.sort(historical_window, axis=0)
    var_index = int(np.ceil((1 - alpha) * sorted_returns.shape[0])) - 1
    VaR_95 = sorted_returns[var_index]
    CVaR_95 = np.mean(sorted_returns[:var_index], axis=0)
    
    w = cp.Variable(n_assets)

    portfolio_CVaR = w.T @ CVaR_95
    objective = cp.Minimize(gamma * portfolio_CVaR - (1 - gamma) * returns)
    constraints = [cp.sum(w) == 1, w >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    optimal_weights_mean_CVaR[i] = w.value

dates = data.index[window_size - 1:]
mean_CVaR_weight = pd.DataFrame(optimal_weights_mean_CVaR, index=dates, columns=data.columns)

output_excel_path = 'D:/2023semester/Lund University/thesis/optimization_mean_CVaR.xlsx'
mean_CVaR_weight.to_excel(output_excel_path)
print("Successful: mean_CVaR")
#############################################################################################
#############################################################################################
print(mean_variance_weight.shape)
print(min_variance_weight.shape)
print(Sharpe_Ratio_weight.shape)
print(Sortino_Ratio_weight.shape)
print(min_correlation_weight.shape)
print(mean_CVaR_weight.shape)
#############################################################################################
#############################################################################################
# average weights
average_mean_variance = mean_variance_weight.mean(axis=0)
average_min_variance = min_variance_weight.mean(axis=0)
average_sharpe_ratio = Sharpe_Ratio_weight.mean(axis=0)
average_sortino_ratio = Sortino_Ratio_weight.mean(axis=0)
average_min_correlation = min_correlation_weight.mean(axis=0)
average_mean_CVaR = mean_CVaR_weight.mean(axis=0)

average_weights_df = pd.DataFrame({
    'Mean-Variance': average_mean_variance,
    'Min-Variance': average_min_variance,
    'Min-Correlation': average_min_correlation,
    'Sharpe Ratio': average_sharpe_ratio,
    'Sortino Ratio': average_sortino_ratio,
    'Mean-CVaR': average_mean_CVaR
})

equal_weights = pd.Series([1/3]*len(average_mean_variance), index=average_mean_variance.index)
average_weights_df.insert(0, 'Equal Weight', equal_weights)

output_excel_path = 'D:/2023semester/Lund University/thesis/average_weights.xlsx'
average_weights_df.to_excel(output_excel_path, float_format='%.6f')
print("Successful: average weights")
#############################################################################################
#############################################################################################
# portfolio performance
mean_CVaR_weight = mean_CVaR_weight.reindex(data.index, fill_value=np.nan)
equal_weight_weight = pd.read_excel('D:/2023semester/Lund University/thesis/optimization_equal_weight.xlsx', index_col=0)
weights = {
    "Equal Weight": equal_weight_weight,
    "Mean-Variance": mean_variance_weight,
    "Min Variance": min_variance_weight,
    "Min Correlation": min_correlation_weight,
    "Sharpe Ratio": Sharpe_Ratio_weight,
    "Sortino Ratio": Sortino_Ratio_weight,
    "Mean-CVaR": mean_CVaR_weight   }


portfolio_returns_dict = {}
for weight_name, weight_data in weights.items():
    returns = np.sum(weight_data.values * data.values, axis=1) 
    portfolio_returns_dict[weight_name] = pd.Series(returns, index=data.index)

risk_free_rate = 0.0002
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

stats_df = pd.DataFrame(portfolio_stats)
output_filepath = 'D:/2023semester/Lund University/thesis/portfolio_performance.xlsx'
stats_df.to_excel(output_filepath, float_format='%.6f')
print('Successful: Performance')




