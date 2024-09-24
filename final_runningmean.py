# Scatter Plot for Suicide Mortality (2000-2019)
# Comparing Genders and Predicting Future Trends
# Written by Lily Gates
# July 2024

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
import rm  # Running mean function that the professor made
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

#######################################
################ SEXES ################
#######################################

# SAVE SEXES DATA
sexes = ["TOTAL", "FEMALE", "MALE"]

ss_sexes = {}
for sex in sexes:
    s_sex = data_original[(data_original["DIM_SEX"] == sex)][["DIM_TIME", "RATE_PER_100000_N"]].groupby(["DIM_TIME"]).mean()
    #print(f"{sex}:")
    #print(s_sex)
    ss_sexes[sex] = [s_sex.index.to_numpy(), s_sex.values.ravel()]


# SETTING UP THE FIGURE/PLOTS -- SEXES
fig1 = plt.figure(figsize=[16,12])
ax1 = plt.subplot2grid((15,2), (0,0), rowspan=6, colspan=2)  # TOTAL, FEMALE, AND MALE
#plt.setp(ax2.get_xticklabels(), visible=False) 
ax2 = plt.subplot2grid((15,2), (8,0), rowspan=3, colspan=2)  # TOTAL
ax3 = plt.subplot2grid((15,2), (12,0), rowspan=3)  # FEMALE
ax4 = plt.subplot2grid((15,2), (12,1), rowspan=3)  # MALE

# PLOTTING  DATA
# Setting Axis Limits
# Y - Axis
ax1.set_ylim(0, 32)  # All
ax2.set_ylim(9, 19)  # Total Sexes
ax3.set_ylim(4, 9)  # Female Sexes
ax4.set_ylim(14, 32)  # Male Sexes

# X - Axis
### Limit the X axis to only show 2000 - 2019
ax1.set_xlim(1999, 2020)
ax2.set_xlim(1999, 2020)
ax3.set_xlim(1999, 2020)
ax4.set_xlim(1999, 2020)
plt.xticks(np.arange(2000, 2020, step=1))
xticks = np.arange(2000, 2020, 1)
xlabels = ["'00", "'01", "'02", "'03", "'04", "'05", "'06", "'07", "'08", "'09", "'10", "'11", "'12", "'13", "'14", "'15", "'16", "'17", "'18", "'19"]


# SETTING TICK LABELS
ax1.set_xlabel('Years')
ax1.set_xticks(xticks, labels=xticks)
ax2.set_xticks(xticks, labels=xticks)
ax3.set_xticks(xticks, labels=xlabels)
ax4.set_xticks(xticks, labels=xlabels)

# Finding Max and Min Values for Rates
# Sexes
max_total_sex = max(ss_sexes["TOTAL"][1])
min_total_sex = min(ss_sexes["TOTAL"][1])
max_f_sex = max(ss_sexes["FEMALE"][1])
min_f_sex = min(ss_sexes["FEMALE"][1])
max_m_sex = max(ss_sexes["MALE"][1])
min_m_sex = min(ss_sexes["MALE"][1])


# PLOTTING DATA
# SEXES: Total, All Sexes
'''
ax1.scatter(x=ss_sexes["TOTAL"][0], y=ss_sexes["TOTAL"][1], c='k', alpha=0.3, label="Total")
ax1.scatter(x=ss_sexes["FEMALE"][0], y=ss_sexes["FEMALE"][1], c='crimson', alpha=0.3, label="Female")
ax1.scatter(x=ss_sexes["MALE"][0], y=ss_sexes["MALE"][1], c='blue', alpha=0.3, label="Male")
# Individual
ax2.scatter(x=ss_sexes["TOTAL"][0], y=ss_sexes["TOTAL"][1], c='k', alpha=0.3, label="Total")
ax3.scatter(x=ss_sexes["FEMALE"][0], y=ss_sexes["FEMALE"][1], c='crimson', alpha=0.3, label="Female")
ax4.scatter(x=ss_sexes["MALE"][0], y=ss_sexes["MALE"][1], c='blue', alpha=0.3, label="Male")
'''

ax1.plot(ss_sexes["TOTAL"][0], ss_sexes["TOTAL"][1], marker='o', c='k', alpha=0.3, label="Total")
ax1.plot(ss_sexes["FEMALE"][0], ss_sexes["FEMALE"][1], marker='o', c='crimson', alpha=0.3, label="Female")
ax1.plot(ss_sexes["MALE"][0], ss_sexes["MALE"][1], marker='o', c='blue', alpha=0.3, label="Male")
# Individual
ax2.plot(ss_sexes["TOTAL"][0], ss_sexes["TOTAL"][1], marker='o', c='k', alpha=0.3, label="Total")
ax3.plot(ss_sexes["FEMALE"][0], ss_sexes["FEMALE"][1], marker='o', c='crimson', alpha=0.3, label="Female")
ax4.plot(ss_sexes["MALE"][0], ss_sexes["MALE"][1], marker='o', c='blue', alpha=0.3, label="Male")

# RUNNING MEAN AND LINEAR REGRESSION (PREDICTION)
# Running Mean (3 Year) # Doing 3 year because the data is rather small
## Generate Running Mean
rm_rate = rm.running_mean((ss_sexes["TOTAL"][1]), 5)  # Using the Prof. function
# Cannot do running mean on first and last 2 values, removes the 0 tails
ax2.plot(ss_sexes["TOTAL"][0][2:-2], ss_sexes["TOTAL"][1][2:-2], c='black', linestyle='-', label='5-Yr Running Mean') 

rm_rate = rm.running_mean((ss_sexes["FEMALE"][1]), 5)  # Using the Prof. function
# Cannot do running mean on first and last 2 values, removes the 0 tails
ax3.plot(ss_sexes["FEMALE"][0][2:-2], ss_sexes["FEMALE"][1][2:-2], c='black', linestyle='-', label='5-Yr Running Mean') 

rm_rate = rm.running_mean((ss_sexes["MALE"][1]), 5)  # Using the Prof. function
# Cannot do running mean on first and last 2 values, removes the 0 tails
ax4.plot(ss_sexes["MALE"][0][2:-2], ss_sexes["MALE"][1][2:-2], c='black', linestyle='-', label='5-Yr Running Mean') 


'''
# Linear Fit
## Generate Linear Fit Coef
coef_lin = np.polyfit(year, temp, 1)  # x value, y value, 1st order
## Generate Linear Fit Formula
lf = np.poly1d(coef_lin)
## Need New X Values
xfit = np.linspace(year[0], year[-1], 200)
## Usig lf function Generate new y values
yfit_1 = lf(xfit)

plt.plot(xfit, yfit_1, color='red', label='Linear Fit')  # Red, line, labeled
print("Linear Fit Formula:")
print(lf)

# FUTURE PROJECTIONS -- from 2023 to 2050 (27 years)
# Linear Fit (to 2050)
xfit_1f = np.linspace(year[0], year[-1]+27, 200)
yfit_1f = lf(xfit_1f)
plt.plot(xfit_1f, yfit_1f, ls='--', color='red', label='Linear Fit Projection to 2050')  # Red, dashed line, labeled


# Linear Fit
## Generate Linear Fit Coef
# x value, y value, 1st order
print(ss_sexes["TOTAL"][0])
print(ss_sexes["TOTAL"][1])
coef_lin = np.polyfit(ss_sexes["TOTAL"][0], ss_sexes["TOTAL"][1], 1)[1]
print(coef_lin)

## Generate Linear Fit Formula
lf = np.poly1d(coef_lin)

## Need New X Values
ss_sexes["TOTAL"][0][2:-2]
xfit = np.linspace(ss_sexes["TOTAL"][0][0], ss_sexes["TOTAL"][0][-1], 200)
## Usig lf function Generate new y values
yfit_1 = lf(xfit)

ax1.plot(xfit, yfit_1, color='magenta', label='Linear Fit')  # Red, line, labeled
#print("Linear Fit Formula:")
#print(lf)

# FUTURE PROJECTIONS -- from 2020 to 2050 (30 years)
# Linear Fit (to 2050)
xfit_1f = np.linspace(ss_sexes["TOTAL"][0][0], ss_sexes["TOTAL"][0][-1]+30, 200)
yfit_1f = lf(xfit_1f)
ax1.plot(xfit_1f, yfit_1f, ls='--', color='rebeccapurple', label='Linear Fit Projection to 2050') 
'''

# TITLES AND AXIS LABELS
# Legend
#ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 0.2), fancybox=True, shadow=True, ncol=7)
ax1.legend(loc=0)
ax2.legend(loc=0)
ax3.legend(loc=0)
ax4.legend(loc=0)

# Axis 1 (Global)
ax1.set_title('Annual Suicide Mortality Rates by Sex')
ax1.set_ylabel('Suicide Rate\nper 100,000 people')
ax1.set_xlabel('Years') 

# Axis 2-4 ('Suicide Rate\n(per 100,000 people)')
ax2.set_title('\n5-Year Running Mean for Suicide Mortality Rates')
ax2.set_ylabel('Suicide Rate\nper 100,000 people')
ax3.set_ylabel('Suicide Rate\nper 100,000 people')
ax4.set_ylabel('Suicide Rate\nper 100,000 people')

#Figure Title
fig1.suptitle('Comparing Annual Suicide Mortality Rates (2000-2019)\nBy Sexes', fontsize=16, ha='center')
# Footer
fig1.text(0.5, 0.025, "Source: World Health Organization 2024 data.who.int, Suicide mortality rate (per 100 000 population) [Indicator]. https://data.who.int/indicators/i/F08B4FD/16BBF41 (Accessed on 1 July 2024)\nVisualized by Lily Gates for AOSC247 at the University of Maryland", fontsize=7, ha='center')

plt.show()


