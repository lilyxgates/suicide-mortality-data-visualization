# Ladder Plot for Suicide Mortality (2000-2019)
# Comparing Incomes
# Generating MLR
# Written by Lily Gates
# July 2024

# Data file: suicide_mortality.csv
# Data Columns from Original Data File:
### 4 Year
### 7 Region -- (Country, WHO Region, Global, World Bank Income Group)
### 10 Geographic Name -- (Country Names, Regions[6] (Africa, Americas, Eastern Mediterranean, Europe, South-East Asia, Western Pacific), Global (World), Income[4] (Low, Lower-Middle, Upper-Middle, High))
### 11 Sex[3] -- (Total, Male Female)
### 12 Age[9] -- (Total, 15-24, 25-34, 35-44, 45-54, 55-64, 65-74, 75-84, 85+)
### 13 Rate per 100,000
### 14 Rate per 100,000 -- (Lower Limit)
### 15 Rate per 100,000 -- (Upper Limit)

# IMPORT MODULES (in ABC order)
import cartopy
import cartopy.crs as ccrs
import datetime as dt
import glob
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import statsmodels.api as sm
import time

from cartopy.mpl.ticker import LatitudeFormatter
from cartopy.mpl.ticker import LongitudeFormatter


# READING IN THE FILE(S)
os.chdir('/homes/metogra/lilgates/AOSC247/Week6/FinalProject/csv_data')

# Title: Annual Suicide Mortality (2000-2019) -- ORIGINAL
data_original = pd.read_csv('suicide_mortality_original.csv', delim_whitespace=False, skiprows=0)
#print("ORIGINAL")
#print(data_original)

### 4 Year
### 7 Region -- (Country, WHO Region, Global, World Bank Income Group)
### 10 Geographic Name -- (Country Names, Regions[6] (Africa, Americas, Eastern Mediterranean, Europe, South-East Asia, Western Pacific), Global (World), Income[4] (Low, Lower-Middle, Upper-Middle, High))
### 11 Sex[3] -- (Total, Male Female)
### 12 Age[9] -- (Total, 15-24, 25-34, 35-44, 45-54, 55-64, 65-74, 75-84, 85+)
### 13 Rate per 100,000

# ASSIGN VARIABLES
#years = original_data[:, 4]   # x-axis for all graphs





# SAVE INCOME DATA
incomes = ["World", "High-income economies", "Low-income economies", "Lower-middle-income economies", "Upper-middle-income economies"]

ss_incomes = {}

for income in incomes:
    df_income = data_original[(data_original["GEO_NAME_SHORT"] == income) & (data_original["DIM_SEX"] == "TOTAL")]
    #print(df_income)

    s_income = pd.Series(data=df_income["RATE_PER_100000_N"].to_list(), index=df_income["DIM_TIME"].to_list())
    #print(s_income)
    ss_incomes[income] = s_income

#print(ss_incomes)


# EXTRACT VARIABLES
year = ???
obs_world = ss_incomes["World"]
low = ss_incomes["Low-income economies"]
lowmid = ss_incomes["Lower-middle-income economies"]
upmid = ss_incomes["Upper-middle-income economies"]
high = ss_incomes["High-income economies"]

# Defines the number of observations and regressors 
nobs = len(year)  # Number of observations
nrgs = 5  # Number of regressors + 1 (1 is for the y-intercept const., low, lowmid, upmid, high) 

# Create a matrix of 1s with the dimentions defined above
regressors = np.ones((nobs, nrgs), dtype=np.float64)  # Could be np.float

# Populate the matrix columns with regressors (leaving column 0 as 1s) 
regressors[:,1] = low
regressors[:,2] = lowmid
regressors[:,3] = upmid
regressors[:,4] = high

# Create an ordinary least squares model
model = sm.OLS(obs_world, regressors)  # (known values, regressors)

# Extract the constants
constants = model.fit().params

print(model.fit().summary())  # Prints the model, the constants, the fit etc.
print(constants)

# Calculate the model fit result
model_world = constants[0] + constants[1]*low + constants[2]*lowmid + constants[3]*upmid + constants[4]*high  # Creates the y intercepts

# Plot
plt.plot(year, obs_world, c='k', label = 'Observed') # Observations
plt.plot(year, model_world, c='r', lw=1, label = 'MLR w/ Python') # MLR with Python

"""

# PLOTTING REGIONAL DATA
fig1 = plt.figure(figsize=[15,8])

ax1 = plt.subplot2grid((12,2), (0,0), rowspan=5, colspan=2)
#plt.setp(ax2.get_xticklabels(), visible=False)

ax2 = plt.subplot2grid((12,2), (7,0), rowspan=2)
ax3 = plt.subplot2grid((12,2), (7,1), rowspan=2)
ax4 = plt.subplot2grid((12,2), (10,0), rowspan=2)
ax5 = plt.subplot2grid((12,2), (10,1), rowspan=2)

# PLOTTING REGIONAL DATA


#Finding Min and Max Values
min_low = min(ss_incomes["Low-income economies"])
max_low = max(ss_incomes["Low-income economies"])

min_lowmid = min(ss_incomes["Lower-middle-income economies"])
max_lowmid = max(ss_incomes["Lower-middle-income economies"])

min_upmid = min(ss_incomes["Upper-middle-income economies"])
max_upmid = max(ss_incomes["Upper-middle-income economies"])

min_high = min(ss_incomes["High-income economies"])
max_high = max(ss_incomes["High-income economies"])

# SET Y-AXIS RANGES

# True Max and Min Set
#ax2.set_ylim(min_low, max_low)
#ax3.set_ylim(min_lowmid, max_lowmid)
#ax4.set_ylim(min_upmid, max_upmid)
#ax5.set_ylim(min_high, max_high)

# Semi Manually Set
# Setting Axis Limits
ymin = 5
ymax = 16
ax1.set_ylim(ymin, ymax)
ax2.set_ylim(ymin, ymax)
ax3.set_ylim(ymin, ymax)
ax4.set_ylim(ymin, ymax)
ax5.set_ylim(ymin, ymax)

# Manually Set
ax1.set_ylim(4, 16)
#ax2.set_ylim(5, 9)
#ax3.set_ylim(7, 15)
#ax4.set_ylim(5, 15)
#ax5.set_ylim(12, 15)
# Set Using True Max and Min

# Global and ALL Incomes
#ax1.plot(ss_incomes["World"], c='purple', linestyle = "--", linewidth = '1', label="Calculated Model")  # World
ax1.plot(ss_incomes["World"], c='k', linestyle = "--", linewidth = '2', label="Global")  # World
ax1.plot(ss_incomes["Low-income economies"], c='red', linewidth = '1.5', label="Low")
ax1.plot(ss_incomes["Lower-middle-income economies"], c='darkorange', linewidth = '1.5', label="Lower-Middle")
ax1.plot(ss_incomes["Upper-middle-income economies"], linewidth = '1.5', c='darkgreen', label="Upper-Middle")
ax1.plot(ss_incomes["High-income economies"], c='darkblue', linewidth = '1.5', label="High")

# Incomes
ax2.plot(ss_incomes["Low-income economies"], c='red', label="Low-income economies")
ax3.plot(ss_incomes["Lower-middle-income economies"],c='darkorange', label="Lower-middle-income economies")
ax4.plot(ss_incomes["Upper-middle-income economies"], c='darkgreen', label="Upper-middle-income economies")
ax5.plot(ss_incomes["High-income economies"], c='darkblue', label="High-income economies")


# TITLES AND AXIS LABELS
#Figure Title
fig1.suptitle('Comparing Annual Suicide Mortality Rates (2000-2019)', fontsize=16, ha='center')
# Footer
fig1.text(0.5, 0.025, "Source: World Health Organization 2024 data.who.int, Suicide mortality rate (per 100 000 population) [Indicator]. https://data.who.int/indicators/i/F08B4FD/16BBF41 (Accessed on 1 July 2024)\nVisualized by Lily Gates for AOSC247 at the University of Maryland", fontsize=7, ha='center')

# X-Axis
plt.xticks(np.arange(2000, 2020, step=1))

xticks = np.arange(2000, 2020, 1)
xlabels = ["'00", "'01", "'02", "'03", "'04", "'05", "'06", "'07", "'08", "'09", "'10", "'11", "'12", "'13", "'14", "'15", "'16", "'17", "'18", "'19"]

ax1.set_xticks(xticks, labels=xticks)
ax1.set_xlabel('Years')

ax2.set_xticks(xticks, labels=xlabels)
ax3.set_xticks(xticks, labels=xlabels)
ax4.set_xticks(xticks, labels=xlabels)
ax5.set_xticks(xticks, labels=xlabels)


# Legend
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 0.15), fancybox=True, shadow=True, ncol=5)

# Axis 1 (Global)
ax1.set_title('Annual Suicide Mortality Rates of Countries with Varying Income-Level Economies\n(Income Levels Catagorized by The Work Bank)\n')
ax1.set_ylabel('Suicide Rate\nper 100,000 people')
ax1.set_xlabel('Years') 

# Axis 2-7 ('Suicide Rate\n(per 100,000 people)')
ax2.set_ylabel('Low\nIncome')
ax3.set_ylabel('Lower-Mid\nIncome')
ax4.set_ylabel('Upper-Mid\nIncome')
ax5.set_ylabel('High\nIncome')

plt.show()

"""

