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
from sstudentt import SST
from scipy.optimize import minimize
import cvxpy as cp

def vecl(matrix):  #get lower matrix
    lower_matrix = np.tril(matrix,k=-1)
    array_with_zero = np.matrix(lower_matrix).A1
    array_without_zero = array_with_zero[array_with_zero!=0]
    return array_without_zero


def garch_skewt_to_u(rets, res):# parameters
    mu = res.params['mu']
    eta = res.params['eta']  #degrees of freedom
    nu=eta
    tau = res.params['lambda']
    est_r = rets - mu
    mu = res.params['mu']
    eta = res.params['eta']
    nu = eta
    sigma = np.sqrt(res.conditional_volatility**2)
    tau = res.params['lambda']
    if tau >3:
        tau=tau
    else:
        tau=3
    dist = SST(mu, sigma, nu, tau)
    std_res = res.std_resid
    udata = dist.p(std_res)
    return mu, udata

def sst(rets, res):#skewed student t distribution
    mu = res.params['mu']
    eta = res.params['eta']  
    nu = eta
    sigma = np.sqrt(res.conditional_volatility**2)
    tau = res.params['lambda']
    if tau >3:
        tau=tau
    else:
        tau=3
    dist = SST(mu, sigma, nu, tau)
    return dist

    std_res = est_r / h
    if tau > 2:
        tau = tau
    else:
        tau = 3
    if nu > 0:
        nu = nu
    else:
        nu = 5
    dist = SST(0, 1, nu, tau)
    udata
    udata = udata[~np.isnan(udata)]
    udata = [arr for arr in udata if arr.size > 0]
    return udata, dist

def loglike_skewt_dcc(theta, udata):#loglikelihood
    N, T = np.shape(udata)
    llf = np.zeros((T,1))
    dist =  sst(rets, res)
    trdata = np.array(dist.q(udata).T, ndmin=2) #(6329, 14)
    Rt, veclRt, Qt =  dcceq(theta,trdata)
        
    for i in range(0,T):
        llf[i] = -0.5* np.log(np.linalg.det(Rt[:,:,i]))
        llf[i] = llf[i] - 0.5 *  np.matmul(np.matmul(trdata[i,:] , (np.linalg.inv(Rt[:,:,i]) - np.eye(N))) ,trdata[i,:].T)
    llf = np.sum(llf)
    return -llf

def dcceq(theta, trdata):#conditional correlation matrix
    T, N = np.shape(trdata)
    a, b = theta
    if min(a, b) < 0 or max(a, b) > 1 or a + b > .999999:
        a = .9999 - b
    Qt = np.zeros((N, N, T))
    Qt[:,:,0] = np.cov(trdata.T)
    Rt = np.zeros((N, N, T))
    veclRt = np.zeros((T, int(N*(N-1)/2)))    
    Rt[:,:,0] = np.corrcoef(trdata.T)
    for j in range(1,T):
        Qt[:,:,j] = Qt[:,:,0] * (1-a-b)
        Qt[:,:,j] = Qt[:,:,j] + a * np.matmul(trdata[[j-1]].T, trdata[[j-1]])
        Qt[:,:,j] = Qt[:,:,j] + b * Qt[:,:,j-1]
        Rt[:,:,j] = np.divide(Qt[:,:,j] , np.matmul(np.sqrt(np.array(np.diag(Qt[:,:,j]), ndmin=2)).T , np.sqrt(np.array(np.diag(Qt[:,:,j]), ndmin=2))))
    for j in range(0,T):
        veclRt[j, :] = vecl(Rt[:,:,j].T)
    return Rt, veclRt, Qt

model_parameters = {}
udata_list = []
mu_values = {}
def run_garch_on_return(rets, udata_list, model_parameters):
    for x in rets.columns:
        am = arch_model(rets[x],vol='Garch', p=1, o=0, q=1, dist = 'skewt')
        short_name = x.split()[0]
        model_parameters[short_name] = am.fit(disp='off')
        mu, udata = garch_skewt_to_u(rets[x], model_parameters[short_name])
        udata_list.append(udata)
        mu_values[short_name] = mu
    return mu_values, udata_list, model_parameters

data = pd.read_excel('D:/2023semester/Lund University/thesis/Data.xlsx')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data_scaled = data * 100
rets = data_scaled

garch_results = {}
params_dict = {}
udata_list = []
model_parameters = {}
mu_values, udata_list, model_parameters = run_garch_on_return(rets.iloc[:,:14].dropna(), udata_list, model_parameters)

model_stats = {
    'Model': [],
    'AIC': [],
    'BIC': [],
    'Log-Likelihood': [],
    'LBQ Statistic': [],
    'LBQ p-value': [],
    'Engle\'s LM Statistic': [],
    'Engle\'s LM p-value': []
}
for x in rets.columns:
    am = arch_model(rets[x], vol='Garch', p=1, o=0, q=1, dist='skewt')
    res = am.fit(disp='off')
    residuals = res.resid

    # LBQ test
    lbq_results = acorr_ljungbox(residuals.dropna(), lags=15, return_df=True)
    lbq_stat = lbq_results['lb_stat'].iloc[0]
    lbq_p_value = lbq_results['lb_pvalue'].iloc[0]
    
    # LM test
    lm_stat, lm_p_value, _, _ = het_arch(residuals.dropna())

    # Collect results
    model_stats['Model'].append(x)
    model_stats['AIC'].append(res.aic)
    model_stats['BIC'].append(res.bic)
    model_stats['Log-Likelihood'].append(res.loglikelihood)
    model_stats['LBQ Statistic'].append(lbq_stat)
    model_stats['LBQ p-value'].append(lbq_p_value)
    model_stats['Engle\'s LM Statistic'].append(lm_stat)
    model_stats['Engle\'s LM p-value'].append(lm_p_value)

stats_df = pd.DataFrame(model_stats)

# output_excel_path = 'D:/2023semester/Lund University/thesis/uni_garch.xlsx'
# stats_df.to_excel(output_excel_path, index=False)

cons = ({'type': 'ineq', 'fun': lambda x:  -x[0]  -x[1] +1})
bnds = ((0, 0.5), (0, 0.9997))

opt_out = minimize(loglike_skewt_dcc, [0.01, 0.95], args=(udata_list,), bounds=bnds, constraints=cons)
print(f"Optimization Successful: {opt_out.success}")
print(f"Optimized Parameters: {opt_out.x}")   #[f'{param:.3f}' for param in opt_out.x]
llf  = loglike_skewt_dcc(opt_out.x, udata_list)
print(f"Log-Likelihood: {llf}")

dist = sst(rets, res)
trdata = np.array(dist.p(udata_list).T, ndmin=2)
Rt, veclRt, Qt = dcceq(opt_out.x, trdata)


commodity_names = [x.split()[0] for x in rets.iloc[:,:14].columns]

corr_name_list = []
for i, name_a in enumerate(commodity_names):
    if i == 0:
        pass
    else:
        for name_b in commodity_names[:i]:
            corr_name_list.append(name_a + "-" + name_b)

dcc_corr = pd.DataFrame(veclRt, index = rets.iloc[:,:14].dropna().index, columns= corr_name_list)
# dcc_corr.to_excel('D:/2023semester/Lund University/thesis/time_varying_correlation.xlsx')

########################################average DCC
# average_dcc = dcc_corr.mean()
# num_commodities = 14
# commodity_names = [x.split()[0] for x in rets.iloc[:, :14].columns]
# matrix_df = pd.DataFrame(index=commodity_names, columns=commodity_names)
# for name in average_dcc.index:
#     commodities = name.split('-')
#     commodity_a = commodities[0]
#     commodity_b = commodities[1]
#     matrix_df.loc[commodity_a, commodity_b] = average_dcc[name]
#     matrix_df.loc[commodity_b, commodity_a] = average_dcc[name]
# np.fill_diagonal(matrix_df.values, 1)
# matrix_df = matrix_df.apply(pd.to_numeric)
# matrix_excel_path = 'D:/2023semester/Lund University/thesis/average_dcc_matrix.xlsx'
# matrix_df.to_excel(matrix_excel_path)
# print("Successful: Average DCC matrix")
#############################################################################################
#####################################################################
#Optimization
# print(Qt.shape)  #(14, 14, 6329)
# print(Rt.shape)  #(14, 14, 6329)
# print(mu_values)


################################################mean-varaince 
################################################mean-varaince 
dates = data.index
n_assets = 14
T=6329 
gamma = 1 
mu = np.array(list(mu_values.values()))
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
mean_varaince_weight = optimal_weights_time_varying.T
mean_varaince_weight = pd.DataFrame(mean_varaince_weight, index=dates, columns=commodity_names)
output_excel_path = 'D:/2023semester/Lund University/thesis/optimization_mean_variance.xlsx'
mean_varaince_weight.to_excel(output_excel_path)
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

min_varaince_weight = pd.DataFrame(optimal_weights_min_variance_transposed, index=dates, columns=commodity_names)
output_excel_path = 'D:/2023semester/Lund University/thesis/optimization_min_variance.xlsx'
min_varaince_weight.to_excel(output_excel_path)
print("Successful: Min-Variance")
######################################################################################
######################################################################################

##Sharpe ratio

Rf = 0.02

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
#############################################################################################333
#############################################################################################333
print(mean_varaince_weight.shape)
print(min_varaince_weight.shape)
print(Sharpe_Ratio_weight.shape)
print(Sortino_Ratio_weight.shape)
print(min_correlation_weight.shape)
print(mean_CVaR_weight.shape)





