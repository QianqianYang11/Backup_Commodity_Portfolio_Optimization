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
