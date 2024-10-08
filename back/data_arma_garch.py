import numpy as np
import pandas as pd
import mgarch
import armagarch as ag
import pandas_datareader as web
import matplotlib.pyplot as plt
from RegscorePy import aic, bic
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA

data = pd.read_excel('D:/2023semester/Lund University/thesis/Data.xlsx')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

commodities = [
    "Gold", "Silver", "Palladium", "Platinum",
    "Copper", "WTI Crude Oil", "Brent Crude Oil",
    "Natural Gas", "Corn", "Cocoa", "Cotton",
    "Coffee", "Lean Hogs", "Soybeans"
]

results_summary = {}

for commodity in commodities:
    print(f"Processing {commodity}...")
    returns = data[commodity].dropna()

    best_aic = np.inf
    best_order = None
    for p in range(4):
        for q in range(4):
            try:
                model = ARIMA(returns, order=(p, 0, q))
                results = model.fit()
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = (p, q)
            except:
                continue

    print(f"Best ARMA Order for {commodity}: {best_order} with AIC: {best_aic}")
    
    if best_order:
        p, q = best_order
        arma_garch = arch_model(returns, mean='ARX', lags=p, vol='GARCH', p=1, q=1)
        res = arma_garch.fit(update_freq=5)
        results_summary[commodity] = res.summary()
        print(f"{commodity} GARCH(1,1) Model Summary:\n{res.summary()}\n")
print(results_summary)
results_summary.to_excel('D:/2023semester/Lund University/thesis/arma_garch.xlsx')
