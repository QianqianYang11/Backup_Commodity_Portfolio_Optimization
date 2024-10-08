import numpy as np
import pandas as pd
from arch import arch_model
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
import plotly.figure_factory as ff
from arch import arch_model
from scipy.optimize import fmin, minimize
from scipy.stats import t
from scipy.stats import norm
from math import inf
from IPython.display import display
import bs4 as bs
import requests
import datetime
from scipy.stats import t
import time
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from arch import arch_model

#get lower matrix
def vecl(matrix):  
    lower_matrix = np.tril(matrix,k=-1)
    array_with_zero = np.matrix(lower_matrix).A1
    array_without_zero = array_with_zero[array_with_zero!=0]
    return array_without_zero
# parameters
def garch_t_to_u(rets, res):
    mu = res.params['mu']
    eta = res.params['eta']  #degrees of freedom
    est_r = rets - mu
    h = res.conditional_volatility
    std_res = est_r / h
    udata = t.cdf(std_res, eta)
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

    # Ljung-Box 
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
dcc_plot = px.line(dcc_corr, title = 'Dynamic Conditional Correlation plot', width=1000, height=500)
# dcc_plot.show()
# dcc_corr.to_excel('D:/2023semester/Lund University/thesis/dcc_garch.xlsx')

############################average DCC
average_dcc = dcc_corr.mean()
num_commodities = 14
commodity_names = [x.split()[0] for x in rets.iloc[:, :14].columns]
matrix_df = pd.DataFrame(index=commodity_names, columns=commodity_names)
for name in average_dcc.index:
    commodities = name.split('-')
    commodity_a = commodities[0]
    commodity_b = commodities[1]
    matrix_df.loc[commodity_a, commodity_b] = average_dcc[name]
    matrix_df.loc[commodity_b, commodity_a] = average_dcc[name]
np.fill_diagonal(matrix_df.values, 1)
matrix_df = matrix_df.apply(pd.to_numeric)
# matrix_excel_path = 'D:/2023semester/Lund University/thesis/average_dcc_matrix.xlsx'
# matrix_df.to_excel(matrix_excel_path)
# print(f"Average DCC matrix for the entire period has been successfully exported to {matrix_excel_path}")
############################




