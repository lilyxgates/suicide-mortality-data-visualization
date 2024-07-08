# Ladder Plot for Suicide Mortality (2000-2019)
# Comparing Regions
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

# SAVE REGIONAL DATA
regions = ["Africa", "Americas", "Eastern Mediterranean", "Europe", "South-East Asia", "Western Pacific", "World"]

ss_regions = {}

for region in regions:
    df_region = data_original[(data_original["GEO_NAME_SHORT"] == region) & (data_original["DIM_SEX"] == "TOTAL")]
    #print(df_region)

    s_region = pd.Series(data=df_region["RATE_PER_100000_N"].to_list(), index=df_region["DIM_TIME"].to_list())
    #print(s_region)
    ss_regions[region] = s_region

#print(ss_regions)

# PLOTTING REGIONAL DATA
fig1 = plt.figure(figsize=[15,8])

ax1 = plt.subplot2grid((14,2), (0,0), rowspan=5, colspan=2)
#plt.setp(ax2.get_xticklabels(), visible=False)

ax2 = plt.subplot2grid((14,2), (6,0), rowspan=2)
ax3 = plt.subplot2grid((14,2), (6,1), rowspan=2)
ax4 = plt.subplot2grid((14,2), (9,0), rowspan=2)
ax5 = plt.subplot2grid((14,2), (9,1), rowspan=2)
ax6 = plt.subplot2grid((14,2), (12,0), rowspan=2)
ax7 = plt.subplot2grid((14,2), (12,1), rowspan=2)

# PLOTTING REGIONAL DATA

# Setting Axis Limits
ymin = 5
ymax = 25

# Set Axis Ranges
ax1.set_ylim(0, 25)
ax2.set_ylim(ymin, ymax)
ax3.set_ylim(ymin, ymax)
ax4.set_ylim(ymin, ymax)
ax5.set_ylim(ymin, ymax)
ax6.set_ylim(ymin, ymax)
ax7.set_ylim(ymin, ymax)

# Global and ALL Regions
ax1.plot(ss_regions["World"], c='k', linestyle = "--", linewidth = '2', label="Global")  # World
ax1.plot(ss_regions["Africa"], c='red', linewidth = '1', label="Africa")
ax1.plot(ss_regions["Americas"], c='darkorange', linewidth = '1', label="Americas")
ax1.plot(ss_regions["Eastern Mediterranean"], linewidth = '1', c='gold', label="Eastern Mediterranean")
ax1.plot(ss_regions["Europe"], c='darkgreen', linewidth = '1', label="Europe")
ax1.plot(ss_regions["South-East Asia"], c='darkblue', linewidth = '1', label="South-East Asia")
ax1.plot(ss_regions["Western Pacific"], c='purple', linewidth = '1', label="Western Pacific")

# Regions
ax2.plot(ss_regions["Africa"], c='red', label="Africa")
ax3.plot(ss_regions["Americas"],c='darkorange', label="Americas")
ax4.plot(ss_regions["Eastern Mediterranean"], c='gold', label="Eastern Mediterranean")
ax5.plot(ss_regions["Europe"], c='darkgreen', label="Europe")
ax6.plot(ss_regions["South-East Asia"], c='darkblue', label="South-East Asia")
ax7.plot(ss_regions["Western Pacific"], c='purple', label="Western Pacific")


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
ax6.set_xticks(xticks, labels=xlabels)
ax7.set_xticks(xticks, labels=xlabels)


# Legend
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 0.2), fancybox=True, shadow=True, ncol=7)

# Axis 1 (Global)
ax1.set_title('Annual Suicide Mortality Rates in World Health Organization Regions')
ax1.set_ylabel('Suicide Rate\nper 100,000 people')
ax1.set_xlabel('Years') 

# Axis 2-7 ('Suicide Rate\n(per 100,000 people)'Geographical Zones)
ax2.set_ylabel('Africa')
ax3.set_ylabel('Americas')
ax4.set_ylabel('E. Mediter.')
ax5.set_ylabel('Europe')
ax6.set_ylabel('S.E. Asia')
ax7.set_ylabel('W. Pacific')

plt.show()


# SAVE SEXES DATA
sexes = ["TOTAL", "FEMALE", "MALE"]

ss_sexes = {}

for sex in sexes:
    df_sex = data_original[(data_original["DIM_SEX"] == sex)]
    #print(df_sex)

    s_sex = pd.Series(data=df_sex["RATE_PER_100000_N"].to_list(), index=df_sex["DIM_TIME"].to_list())
    #print(s_sex)
    ss_sexes[sex] = s_sex

"""
### INCOME ECONOMIES ###

# HIGH
High-income economies

# LOW
Low-income economies

# LOWER-MIDDLE
Lower-middle-income economies

# UPPER-MIDDLE
Upper-middle-income economies
"""


