import numpy as np
import pandas as pd
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
from scipy.optimize import minimize
import cvxpy as cp


def vecl(matrix):  # get lower matrix
    lower_matrix = np.tril(matrix, k=-1)
    array_with_zero = np.matrix(lower_matrix).A1
    array_without_zero = array_with_zero[array_with_zero != 0]
    return array_without_zero

def garch_t_to_u(rets, res):
    mu = res.params['mu']
    nu = res.params['nu']
    est_r = rets - mu
    sigma = np.sqrt(res.conditional_volatility**2)
    dist = t(df=nu)
    std_res = res.std_resid
    udata = dist.cdf(std_res)
    return mu, udata

def t_dist(rets, res):
    mu = res.params['mu']
    nu = res.params['nu']
    sigma = np.sqrt(res.conditional_volatility**2)
    dist = t(df=nu)
    return dist


def loglike_t_dcc(theta, udata):
    N, T = np.shape(udata)
    dist = t_dist(rets, res)
    trdata = np.array(dist.ppf(udata).T, ndmin=2)
    Rt, veclRt, Qt = dcceq(theta, trdata)
    llf = -0.5 * T * N * np.log(2 * np.pi)
    for i in range(T):
        Ht = Rt[:, :, i] * Qt[:, :, i] * Rt[:, :, i].T  # Ensuring Ht as Rt*Qt*Rt'
        Ht_inv = np.linalg.inv(Ht)
        llf -= 0.5 * np.log(np.linalg.det(Ht))
        llf -= 0.5 * np.dot(trdata[i, :], np.dot(Ht_inv, trdata[i, :].T))
    return llf

def dcceq(theta, trdata):
    T, N = np.shape(trdata)
    a, b = theta
    if min(a, b) < 0 or max(a, b) > 1 or a + b > .999999:
        a = .9999 - b
    Qt = np.zeros((N, N, T))
    Qt[:, :, 0] = np.cov(trdata.T)
    Rt = np.zeros((N, N, T))
    veclRt = np.zeros((T, int(N * (N - 1) / 2)))
    Rt[:, :, 0] = np.corrcoef(trdata.T)
    for j in range(1, T):
        Qt[:, :, j] = Qt[:, :, 0] * (1 - a - b)
        Qt[:, :, j] = Qt[:, :, j] + \
            a * np.matmul(trdata[[j - 1]].T, trdata[[j - 1]])
        Qt[:, :, j] = Qt[:, :, j] + b * Qt[:, :, j - 1]
        Rt[:, :, j] = np.divide(Qt[:, :, j], np.matmul(
            np.sqrt(np.array(np.diag(Qt[:, :, j]), ndmin=2)).T, np.sqrt(np.array(np.diag(Qt[:, :, j]), ndmin=2))))
    for j in range(T):
        veclRt[j, :] = vecl(Rt[:, :, j].T)
    return Rt, veclRt, Qt


model_parameters = {}
udata_list = []
mu_values = {}

def run_garch_on_return(rets, udata_list, model_parameters):
    for x in rets.columns:
        am = arch_model(rets[x], vol='Garch', p=1, o=0, q=1, dist='t')
        short_name = x.split()[0]
        model_parameters[short_name] = am.fit(disp='off')
        mu, udata = garch_t_to_u(rets[x], model_parameters[short_name])
        udata_list.append(udata)
        mu_values[short_name] = mu
    return mu_values, udata_list, model_parameters



data = pd.read_excel('D:/2023semester/Lund University/thesis/Data.xlsx')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data_scaled = data * 100
rets = data_scaled
N, T = np.shape(data)

garch_results = {}
params_dict = {}
udata_list = []
model_parameters = {}
mu_values, udata_list, model_parameters = run_garch_on_return(
    rets.iloc[:, :14].dropna(), udata_list, model_parameters)


model_stats = {
    'Commodity': [],
    'Coef.': [],
    't-Stat': [],
    'p-value': [],
    'Omega': [],
    'Alpha': [],
    'Beta': [],
    'AIC': [],
    'BIC': [],
    'Log-Likelihood': [],
    'LBQ Statistic': [],
    'LBQ p-value': [],
    'LM Statistic': [],
    'LM p-value': []
}
for x in rets.columns:
    am = arch_model(rets[x], vol='Garch', p=1, o=0, q=1, dist='t')
    res = am.fit(disp='off')
    residuals = res.resid

    # LBQ test
    lbq_results = sm.stats.acorr_ljungbox(
        residuals.dropna(), lags=15, return_df=True)
    lbq_stat = lbq_results['lb_stat'].iloc[0]
    lbq_p_value = lbq_results['lb_pvalue'].iloc[0]

    # LM test
    lm_stat, lm_p_value, _, _ = sm.stats.het_arch(residuals.dropna())

    # Collect results
    model_stats['Commodity'].append(x)
    model_stats['Coef.'].append(res.params['mu'])
    model_stats['t-Stat'].append(res.tvalues['mu'])
    model_stats['p-value'].append(res.pvalues['mu'])
    model_stats['Omega'].append(res.params['omega'])
    model_stats['Alpha'].append(res.params['alpha[1]'])
    model_stats['Beta'].append(res.params['beta[1]'])
    model_stats['AIC'].append(res.aic)
    model_stats['BIC'].append(res.bic)
    model_stats['Log-Likelihood'].append(res.loglikelihood)
    model_stats['LBQ Statistic'].append(lbq_stat)
    model_stats['LBQ p-value'].append(lbq_p_value)
    model_stats['LM Statistic'].append(lm_stat)
    model_stats['LM p-value'].append(lm_p_value)

stats_df = pd.DataFrame(model_stats)


model_stats['a'] = []
model_stats['b'] = []
model_stats['Log-Likelihood'] = []

cons = ({'type': 'ineq', 'fun': lambda x:  - x[0] - x[1] + 1})
bnds = ((0, 0.5), (0, 0.9997))

opt_out = minimize(lambda x: -loglike_t_dcc(x, udata_list), [0.01, 0.95], 
                   bounds=bnds, constraints=cons)


if opt_out.success:
    alpha, beta = opt_out.x
    llf = loglike_t_dcc(opt_out.x, udata_list)

    dcc_results = {
        'Commodity': 'DCC',
        'Coef.': None,
        't-Stat': None,
        'p-value': None,
        'Omega': None,
        'Alpha': None,
        'Beta': None,
        'AIC': None,
        'BIC': None,
        'Log-Likelihood': None,
        'LBQ Statistic': None,
        'LBQ p-value': None,
        'LM Statistic': None,
        'LM p-value': None,
        'a': alpha,
        'b': beta,
        'DCC Log-Likelihood': llf
    }
    dcc_df = pd.DataFrame([dcc_results])
    final_stats_df = pd.concat([stats_df, dcc_df], ignore_index=True)

# output_excel_path = 'D:/2023semester/Lund University/thesis/uni_garch.xlsx'
# final_stats_df.to_excel(output_excel_path, index=False)
# print("Successful: DCC-GARCH result")


# Compute AIC and BIC
def compute_aic_bic(llf, num_params, T):
    aic = -2 * llf + 2 * num_params
    bic = -2 * llf + num_params * np.log(T)
    return aic, bic

# Calculate number of parameters
num_params = 0
for model in model_parameters.values():
    num_params += len(model.params)  
    
num_params += 2  # for a and b in DCC model
print(num_params)
num_params = 59
aic, bic = compute_aic_bic(llf, num_params, T)
print("AIC:", aic)
print("BIC:", bic)
print("llf", opt_out.fun)

trdata_list = []

for key, fitted_model in model_parameters.items():
    nu = fitted_model.params['nu']
    dist = t(df=nu)
    udata = udata_list.pop(0)
    trdata = dist.ppf(udata)
    trdata_list.append(trdata)

trdata_array = np.stack(trdata_list).T
Rt, veclRt, Qt = dcceq(opt_out.x, trdata_array)

print(Qt.shape)  #(14, 14, 6329)
print(Rt.shape)  #(14, 14, 6329)
# print(mu_values)
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

commodity_names = ['Gold', 'Silver', 'Palladium', 'Platinum', 'Copper', 'WTI Crude Oil',
                   'Brent Crude Oil', 'Natural Gas', 'Corn', 'Cocoa', 'Cotton', 'Coffee',
                   'Lean Hogs', 'Soybeans']
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


commodity_names = ['Gold', 'Silver', 'Palladium', 'Platinum', 'Copper', 'WTI Crude Oil',
                    'Brent Crude Oil', 'Natural Gas', 'Corn', 'Cocoa', 'Cotton', 'Coffee',
                    'Lean Hogs', 'Soybeans']

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

commodity_names = ['Gold', 'Silver', 'Palladium', 'Platinum', 'Copper', 'WTI Crude Oil',
                    'Brent Crude Oil', 'Natural Gas', 'Corn', 'Cocoa', 'Cotton', 'Coffee',
                    'Lean Hogs', 'Soybeans']
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

commodity_names = ['Gold', 'Silver', 'Palladium', 'Platinum', 'Copper', 'WTI Crude Oil',
                    'Brent Crude Oil', 'Natural Gas', 'Corn', 'Cocoa', 'Cotton', 'Coffee',
                    'Lean Hogs', 'Soybeans']
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


commodity_names = ['Gold', 'Silver', 'Palladium', 'Platinum', 'Copper', 'WTI Crude Oil',
                    'Brent Crude Oil', 'Natural Gas', 'Corn', 'Cocoa', 'Cotton', 'Coffee',
                    'Lean Hogs', 'Soybeans']

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

equal_weights = pd.Series([1/14]*len(average_mean_variance), index=average_mean_variance.index)
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
