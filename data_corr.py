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
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, jarque_bera
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch



data = pd.read_excel('D:/2023semester/Lund University/thesis/Data.xlsx')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
#correlation
plt.figure(figsize=(12, 8))
correlation = data.corr()
heatmap = sns.heatmap(correlation, annot=True, cmap='binary', fmt=".2f")
plt.show()
correlation.to_excel('D:/2023semester/Lund University/thesis/correlation.xlsx')




















