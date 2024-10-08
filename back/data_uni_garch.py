import numpy as np
import pandas as pd
from arch import arch_model
import statsmodels.api as sm
import plotly.express as px
import plotly.figure_factory as ff
from scipy.optimize import fmin, minimize, approx_fprime
from scipy.stats import t, norm
from math import inf
from IPython.display import display
import bs4 as bs
import requests
import datetime
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy.linalg import inv
from numpy.linalg import norm as vector_norm
import time
from sstudentt import SST

#get lower matrix
def vecl(matrix):  
    lower_matrix = np.tril(matrix,k=-1)
    array_with_zero = np.matrix(lower_matrix).A1
    array_without_zero = array_with_zero[array_with_zero!=0]
    return array_without_zero
# parameters
def garch_t_to_u(rets, res):
    mu = res.params['mu']
    eta = res.params['eta']  # Adjusted to use 'eta' for degrees of freedom
    est_r = rets - mu
    h = res.conditional_volatility
    std_res = est_r / h
    udata = t.cdf(std_res, eta)  # Use 'eta' here
    return udata

####
def loglike_norm_dcc_copula(theta, udata):
    N, T = np.shape(udata)
    llf = np.zeros((T,1))
    trdata = np.array(norm.ppf(udata).T, ndmin=2) 
    Rt, veclRt =  dcceq(theta,trdata)
    for i in range(0,T):
        llf[i] = -0.5* np.log(np.linalg.det(Rt[:,:,i]))
        llf[i] = llf[i] - 0.5 *  np.matmul(np.matmul(trdata[i,:] , (np.linalg.inv(Rt[:,:,i]) - np.eye(N))) ,trdata[i,:].T)
    llf = np.sum(llf)
    return -llf
#conditional correlation matrix
def dcceq(theta,trdata):
    T, N = np.shape(trdata)
    a, b = theta
    if min(a,b)<0 or max(a,b)>1 or a+b > .999999:
        a = .9999 - b
    Qt = np.zeros((N, N ,T))
    Qt[:,:,0] = np.cov(trdata.T)
    Rt =  np.zeros((N, N ,T))
    veclRt =  np.zeros((T, int(N*(N-1)/2)))    
    Rt[:,:,0] = np.corrcoef(trdata.T)
    for j in range(1,T):
        Qt[:,:,j] = Qt[:,:,0] * (1-a-b)
        Qt[:,:,j] = Qt[:,:,j] + a * np.matmul(trdata[[j-1]].T, trdata[[j-1]])
        Qt[:,:,j] = Qt[:,:,j] + b * Qt[:,:,j-1]
        Rt[:,:,j] = np.divide(Qt[:,:,j] , np.matmul(np.sqrt(np.array(np.diag(Qt[:,:,j]), ndmin=2)).T , np.sqrt(np.array(np.diag(Qt[:,:,j]), ndmin=2))))
    for j in range(0,T):
        veclRt[j, :] = vecl(Rt[:,:,j].T)
    return Rt, veclRt

model_parameters = {}
udata_list = []
#Uni GARCH
def run_garch_on_return(rets, udata_list, model_parameters):
    for x in rets:
        am = arch_model(rets[x],vol='Garch', p=1, o=0, q=1, dist = 'skewt')
        short_name = x.split()[0]
        model_parameters[short_name] = am.fit(disp='off')
        udata = garch_t_to_u(rets[x], model_parameters[short_name])
        udata_list.append(udata)
    return udata_list, model_parameters


data = pd.read_excel('D:/2023semester/Lund University/thesis/Data.xlsx')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data_scaled = data * 100
rets = data_scaled

garch_results = {}
params_dict = {}
udata_list = []
model_parameters = {}
udata_list, model_parameters = run_garch_on_return(rets.iloc[:,:14].dropna(), udata_list, model_parameters)


# print(f"Model statistics, along with Ljung-Box Q and Engle's LM test results, have been successfully exported to {output_excel_path}")
cons = ({'type': 'ineq', 'fun': lambda x:  -x[0]  -x[1] +1})
bnds = ((0, 0.5), (0, 0.9997))

start_time = time.time()
opt_out = minimize(loglike_norm_dcc_copula, [0.01, 0.95], args=(udata_list,), bounds=bnds, constraints=cons)
end_time = time.time()
print(f"Optimization Successful: {opt_out.success}")
print(f"Optimized Parameters: {opt_out.x}")
print(f"Execution time: {end_time - start_time} seconds")

llf  = loglike_norm_dcc_copula(opt_out.x, udata_list)
print(f"Log-Likelihood Function: {llf}")

trdata = np.array(norm.ppf(udata_list).T, ndmin=2)
Rt, veclRt = dcceq(opt_out.x, trdata)

commodity_names = [x.split()[0] for x in rets.iloc[:,:14].columns]

corr_name_list = []
for i, name_a in enumerate(commodity_names):
    if i == 0:
        pass
    else:
        for name_b in commodity_names[:i]:
            corr_name_list.append(name_a + "-" + name_b)



dcc_corr = pd.DataFrame(veclRt, index = rets.iloc[:,:14].dropna().index, columns= corr_name_list)
# dcc_plot = px.line(dcc_corr, title = 'Dynamic Conditional Correlation plot', width=1000, height=500)
# dcc_plot.show()
# dcc_corr.to_excel('D:/2023semester/Lund University/thesis/dcc_garch.xlsx')


#############print out
results = {
    'Panel A': [],
    'Panel B': [],
    'Panel C': [],
    'Panel D': []
}

for column in rets.columns:
    model = arch_model(rets[column], p=1, o=0, q=1, dist='skewt')
    res = model.fit(disp='off')
    
    # Panel A - Mean Equation ('mu' parameter)
    results['Panel A'].append({
        'Asset': column,
        'Param': 'μ',
        'Coef.': res.params['mu'],
        't-Stat': res.tvalues['mu'],
        'p-value': res.pvalues['mu']
    })
    
    # Panel B - Variance Equation ('omega', 'alpha[1]', 'beta[1]')
    results['Panel B'].append({
        'Asset': column,
        'Omega': res.params['omega'],
        'Alpha': res.params['alpha[1]'],
        'Beta': res.params['beta[1]']
    })

    # Panel D - Diagnostics
    lbq_results = acorr_ljungbox(res.resid.dropna(), lags=15, return_df=True)
    lbq_stat, lbq_pvalue = lbq_results['lb_stat'].iloc[0], lbq_results['lb_pvalue'].iloc[0]
    lm_stat, lm_pvalue, _, _ = het_arch(res.resid.dropna())
    
    results['Panel D'].append({
        'Asset': column,
        'AIC': res.aic,
        'BIC': res.bic,
        'Log-Likelihood': res.loglikelihood,
        'LBQ Stat': lbq_stat,
        'LBQ p-value': lbq_pvalue,
        'Engle\'s LM Stat': lm_stat,
        'Engle\'s LM p-value': lm_pvalue
    })

# Panel C - DCC Parameters 
results['Panel C'].append({
    'Parameter': 'a',
    'Value': opt_out.x[0]
})
results['Panel C'].append({
    'Parameter': 'b',
    'Value': opt_out.x[1]
})


with pd.ExcelWriter('D:/2023semester/Lund University/thesis/uni_garch.xlsx') as writer:
    for panel, data in results.items():
        pd.DataFrame(data).to_excel(writer, sheet_name=panel, index=False)

print("Export completed successfully.")



#################### DCC p-value and t-stat
N = len(udata_list[0])  # Number of observations 

def numerical_hessian(f, theta, eps=1e-5):
    """A simple numerical approximation of the Hessian matrix of f at theta."""
    n = len(theta)
    hessian = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            theta_inc = theta.copy()
            theta_dec = theta.copy()
            theta_inc[i] += eps
            theta_inc[j] += eps
            theta_dec[i] -= eps
            theta_dec[j] -= eps
            hessian[i, j] = (f(theta_inc) - 2 * f(theta) + f(theta_dec)) / eps**2
    return hessian

hessian_approx = numerical_hessian(lambda x: loglike_norm_dcc_copula(x, udata_list), opt_out.x)
var_cov_matrix = inv(hessian_approx)
standard_errors = np.sqrt(np.diag(var_cov_matrix))
t_stats = opt_out.x / standard_errors

p_values = 2 * (1 - t.cdf(np.abs(t_stats), df=N - len(opt_out.x)))

def loglike_norm_dcc_copula_wrapper(theta):
    return loglike_norm_dcc_copula(theta, udata_list)

def estimate_hessian(func, theta, epsilon=1e-5):
    """Estimate the Hessian matrix using finite differences."""
    n = len(theta)
    hessian = np.zeros((n, n))
    f0 = func(theta)  
    for i in range(n):
        theta_plus = np.array(theta, copy=True)
        theta_plus[i] += epsilon
        gradient_plus = approx_fprime(theta_plus, func, epsilon)
        for j in range(n):
            theta_minus = np.array(theta, copy=True)
            theta_minus[j] -= epsilon
            gradient_minus = approx_fprime(theta_minus, func, epsilon)
            hessian[i, j] = (gradient_plus[j] - gradient_minus[j]) / (2 * epsilon)
    return hessian

N = len(udata_list[0])  
hessian_approx = estimate_hessian(loglike_norm_dcc_copula_wrapper, opt_out.x)
var_cov_matrix = inv(hessian_approx)
standard_errors = np.sqrt(np.diag(var_cov_matrix))
t_stats = opt_out.x / standard_errors

results_df = pd.DataFrame({
    'Parameter': ['a', 'b'],  # Adjust this list based on your actual model parameters
    'Estimate': np.round(opt_out.x, 8),
    'Standard Error': np.round(standard_errors, 8),
    'T-Statistic': np.round(t_stats, 8),
    'P-Value': np.round(p_values, 8)
})



# results_df.to_excel('D:/2023semester/Lund University/thesis/test_dcc.xlsx')

p_values = []
for t_stat in t_stats:
    if np.abs(t_stat) > 10:  
        p_value = 2 * (1 - t.cdf(10, df=N - len(opt_out.x)))  # Use a threshold value for extreme t-stats
    else:
        p_value = 2 * (1 - t.cdf(np.abs(t_stat), df=N - len(opt_out.x)))
    p_values.append(p_value)
p_values = np.array(p_values)
p_values = ["{:.8e}".format(p) for p in p_values]

print("P-Values:", p_values)




