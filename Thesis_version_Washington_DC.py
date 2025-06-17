# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 17:17:07 2025

@author: deshp
"""

import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS,
                         summarize)
from ISLP.models import (summarize,
                         poly,
                         ModelSpec as MS)
from statsmodels.stats.anova import anova_lm
from pygam import (s as s_gam,
                   l as l_gam,
                   f as f_gam,
                   LinearGAM,
                   LogisticGAM)

from ISLP.transforms import (BSpline,
                             NaturalSpline)
from ISLP.models import bs, ns
from ISLP.pygam import (approx_lam,
                        degrees_of_freedom,
                        plot as plot_gam,
                        anova as anova_gam)
from ISLP import confusion_table
from ISLP.models import contrast
from sklearn.discriminant_analysis import \
     (LinearDiscriminantAnalysis as LDA,
      QuadraticDiscriminantAnalysis as QDA)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os
from statsmodels.stats.outliers_influence \
 import variance_inflation_factor as VIF
from ISLP.models import sklearn_sm
from statsmodels.stats.anova import anova_lm
from sklearn.model_selection import \
     (cross_validate,
      KFold,
      ShuffleSplit)
from functools import partial
from matplotlib.pyplot import subplots
from statsmodels.api import OLS
import sklearn.model_selection as skm
import sklearn.linear_model as skl
from sklearn.preprocessing import StandardScaler
from ISLP import load_data
from ISLP.models import ModelSpec as MS
from functools import partial

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from ISLP.models import \
     (Stepwise,
      sklearn_selected,
      sklearn_selection_path)
from sklearn.linear_model import LinearRegression
import os
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from sklearn.metrics import pairwise_distances_argmin
from sklearn.cluster import \
     (KMeans,
      AgglomerativeClustering)
from scipy.cluster.hierarchy import \
     (dendrogram,
      cut_tree)
from ISLP.cluster import compute_linkage
from sklearn.decomposition import PCA
from pygam import (s as s_gam,
                   l as l_gam,
                   f as f_gam,
                   LinearGAM,
                   LogisticGAM)

from ISLP.transforms import (BSpline,
                             NaturalSpline)
from ISLP.models import bs, ns
from ISLP.pygam import (approx_lam,
                        degrees_of_freedom,
                        plot as plot_gam,
                        anova as anova_gam)

from pygam import (s as s_gam,
                   l as l_gam,
                   f as f_gam,
                   LinearGAM,
                   LogisticGAM)

from ISLP.transforms import (BSpline,
                             NaturalSpline)
from ISLP.models import bs, ns
from ISLP.pygam import (approx_lam,
                        degrees_of_freedom,
                        plot as plot_gam,
                        anova as anova_gam)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os
os.getcwd()
os.chdir(r'C:\Users\deshp\Desktop\MS ECON\Thesis')



import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load datasets
ridership = pd.read_excel('Daily Ridership Data MTA.xlsx')
gas_prices = pd.read_excel('Gasoline Prices.xlsx')  # Make sure gasoline is CSV
wti_prices = pd.read_excel('Global Crude Oil Prices.xlsx')

# Step 2: Basic cleaning
ridership.columns = [col.strip() for col in ridership.columns]
ridership['Date'] = pd.to_datetime(ridership['Date'])
ridership = ridership.rename(columns={ridership.columns[6]: 'Ridership'})

ridership.dtypes
# Weekly aggregation (week starts Monday)
weekly_ridership = (
    ridership.set_index('Date')
    .resample('W-MON')['Ridership']
    .sum()
    .reset_index()
    .rename(columns={'Date': 'Week_Start', 'Ridership': 'Weekly_Ridership'})
)
gas_prices.columns = ['Date', 'Gas_Price']
wti_prices.columns = ['Date', 'WTI_Price']
gas_prices['Date'] = pd.to_datetime(gas_prices['Date'], errors='coerce')
wti_prices['Date'] = pd.to_datetime(wti_prices['Date'], errors='coerce')

# Align to start of week (Monday)
gas_prices['Week_Start'] = gas_prices['Date'] - pd.to_timedelta(gas_prices['Date'].dt.weekday, unit='D')
wti_prices['Week_Start'] = wti_prices['Date'] - pd.to_timedelta(wti_prices['Date'].dt.weekday, unit='D')

# Weekly averages
weekly_gas = gas_prices.groupby('Week_Start')['Gas_Price'].mean().reset_index()
weekly_oil = wti_prices.groupby('Week_Start')['WTI_Price'].mean().reset_index()

# --------------------------
# Step 3: Merge all datasets
# --------------------------
merged_df = (
    weekly_ridership
    .merge(weekly_gas, on='Week_Start', how='inner')
    .merge(weekly_oil, on='Week_Start', how='inner')
)

# Filter out bad rows
merged_df = merged_df[
    (merged_df['Weekly_Ridership'] > 0) &
    (merged_df['Gas_Price'] > 0) &
    (merged_df['WTI_Price'] > 0)
]

# --------------------------
# Step 4: Log Transformation
# --------------------------
merged_df['log_ridership'] = np.log(merged_df['Weekly_Ridership'])
merged_df['log_gas'] = np.log(merged_df['Gas_Price'])
merged_df['log_oil'] = np.log(merged_df['WTI_Price'])

# Extract Year and Month from Week_Start
merged_df['Year'] = merged_df['Week_Start'].dt.year
merged_df['Month'] = merged_df['Week_Start'].dt.month


import matplotlib.pyplot as plt

# Create a formatted Year-Week label
merged_df['Year_Week_Label'] = 'W' + merged_df['Week_Start'].dt.isocalendar().week.astype(str) + ' ' + merged_df['Year'].astype(str)

# Redo the plot using custom x-axis labels
fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)

# Common x-tick locations and labels (e.g., every 10 weeks)
tick_locs = merged_df.index[::10]
tick_labels = merged_df['Year_Week_Label'].iloc[::10]

# Plot Weekly Ridership
axes[0].plot(merged_df.index, merged_df['Weekly_Ridership'], color='blue')
axes[0].set_title('Weekly Rail Ridership', fontsize=16)
axes[0].set_ylabel('Ridership', fontsize=14)
axes[0].grid(True)

# Plot Gas Prices
axes[1].plot(merged_df.index, merged_df['Gas_Price'], color='green')
axes[1].set_title('Weekly Gasoline Prices', fontsize=16)
axes[1].set_ylabel('Gas Price (USD)', fontsize=14)
axes[1].grid(True)

# Plot Oil Prices
axes[2].plot(merged_df.index, merged_df['WTI_Price'], color='red')
axes[2].set_title('Weekly WTI Crude Oil Prices', fontsize=16)
axes[2].set_ylabel('Oil Price (USD)', fontsize=14)
axes[2].set_xlabel('Week', fontsize=14)
axes[2].grid(True)

# Apply custom x-axis ticks
plt.xticks(ticks=tick_locs, labels=tick_labels, rotation=45)
plt.tight_layout()
plt.show()



merged_df = merged_df[merged_df['Year'] >= 2021]
import matplotlib.pyplot as plt

# Create a formatted Year-Week label
merged_df['Year_Week_Label'] = 'W' + merged_df['Week_Start'].dt.isocalendar().week.astype(str) + ' ' + merged_df['Year'].astype(str)

# Redo the plot using custom x-axis labels
fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)

# Common x-tick locations and labels (e.g., every 10 weeks)
tick_locs = merged_df.index[::10]
tick_labels = merged_df['Year_Week_Label'].iloc[::10]

# Plot Weekly Ridership
axes[0].plot(merged_df.index, merged_df['Weekly_Ridership'], color='blue')
axes[0].set_title('Weekly Rail Ridership', fontsize=16)
axes[0].set_ylabel('Ridership', fontsize=14)
axes[0].grid(True)

# Plot Gas Prices
axes[1].plot(merged_df.index, merged_df['Gas_Price'], color='green')
axes[1].set_title('Weekly Gasoline Prices', fontsize=16)
axes[1].set_ylabel('Gas Price (USD)', fontsize=14)
axes[1].grid(True)

# Plot Oil Prices
axes[2].plot(merged_df.index, merged_df['WTI_Price'], color='red')
axes[2].set_title('Weekly WTI Crude Oil Prices', fontsize=16)
axes[2].set_ylabel('Oil Price (USD)', fontsize=14)
axes[2].set_xlabel('Week', fontsize=14)
axes[2].grid(True)

# Apply custom x-axis ticks
plt.xticks(ticks=tick_locs, labels=tick_labels, rotation=45)
plt.tight_layout()
plt.show()




# Plot 1: Gas Price vs Oil Price (2021 onwards)
x1 = merged_df['WTI_Price']
y1 = merged_df['Gas_Price']
m1, b1 = np.polyfit(x1, y1, 1)
y1_fit = m1 * x1 + b1

plt.figure(figsize=(8, 5))
plt.scatter(x1, y1, alpha=0.7, label='Data Points', color='green')
plt.plot(x1, y1_fit, color='black', label=f'Fit: y = {m1:.2f}x + {b1:.2f}')
plt.title('Gasoline Price vs WTI Oil Price (2021 Onwards)')
plt.xlabel('WTI Oil Price (USD/barrel)')
plt.ylabel('Gasoline Price (USD/gallon)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot 2: Gas Price vs Weekly Ridership (2021 onwards)
x2 = merged_df['Gas_Price']
y2 = merged_df['Weekly_Ridership']
m2, b2 = np.polyfit(x2, y2, 1)
y2_fit = m2 * x2 + b2

plt.figure(figsize=(8, 5))
plt.scatter(x2, y2, alpha=0.7, label='Data Points', color='blue')
plt.plot(x2, y2_fit, color='red', label=f'Fit: y = {m2:.2f}x + {b2:.2f}')
plt.title('Weekly Ridership vs Gasoline Price (2021 Onwards)')
plt.xlabel('Gasoline Price (USD/gallon)')
plt.ylabel('Weekly Ridership')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()




# IV/2SLS Method
# Create dummy variables for both (drop first to avoid dummy variable trap)
year_dummies = pd.get_dummies(merged_df['Year'], prefix='year', dtype=int, drop_first=True)
month_dummies = pd.get_dummies(merged_df['Month'], prefix='month', dtype=int, drop_first=True)

# Concatenate dummies with the original DataFrame
merged_df = pd.concat([merged_df, year_dummies, month_dummies], axis=1)

# Define columns to use as controls
control_cols = list(year_dummies.columns) + list(month_dummies.columns)


# First Stage: log_gas ~ log_oil + year/month controls
X_first = sm.add_constant(merged_df[['log_oil'] + control_cols])
y_first = merged_df['log_gas']
X_first = X_first.apply(pd.to_numeric, errors='coerce').dropna()
y_first = y_first.loc[X_first.index]

first_stage = sm.OLS(y_first, X_first).fit()
first_stage.summary()
merged_df['log_gas_hat'] = first_stage.fittedvalues

# Second Stage: log_ridership ~ predicted log_gas + year/month controls
X_second = sm.add_constant(merged_df[['log_gas_hat'] + control_cols])
X_second = X_second.apply(pd.to_numeric, errors='coerce').dropna()
y_second = merged_df.loc[X_second.index, 'log_ridership']

second_stage = sm.OLS(y_second, X_second).fit()
second_stage.summary()