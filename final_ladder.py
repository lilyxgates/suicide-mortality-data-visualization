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


"""
### INCOME ECONOMIES ###
"High-income economies"
"Low-income economies"
"Lower-middle-income economies"
"Upper-middle-income economies"

"""

'''
#Suicide Rates

fig1 = plt.figure(figsize=[11,14])

#Set Plot Grid
ax1 = plt.subplot2grid((10,1), (0,0), rowspan=3)
plt.setp(ax1.get_xticklabels(), visible=False)

ax2 = plt.subplot2grid((10,1), (4,0), rowspan=2)
plt.setp(ax2.get_xticklabels(), visible=False)

ax3 = plt.subplot2grid((10,1), (6,0), rowspan=2)
plt.setp(ax3.get_xticklabels(), visible=False)

ax4 = plt.subplot2grid((10,1), (8,0), rowspan=2)    # Wanting to show tick marks on the bottom most graph only, helps with readibility and reduce clutter on screen

# Plotting Data
ax1.plot(years, glob, c='k')
ax2.plot(years, nHem, c='b')
ax3.plot(years, sHem, c='r')
ax4.plot(years, equat, c='g')

# Find Out Max and Min Values for Each Plot

# Global
#minTglob = np.min(glob)
#maxTglob = np.max(glob)
#print("Global Temperature Range: {0} to {1}".format(minTglob, maxTglob))

# N. Hemisphere
#minTnHem = np.min(nHem)
#maxTnHem = np.max(nHem)
#print("N. Hemisphere Temperature Range: {0} to {1}".format(minTnHem, maxTnHem))

# S. Hemisphere
#minTsHem = np.min(sHem)
#maxTsHem = np.max(sHem)

#print("S. Hemisphere Temperature Range: {0} to {1}".format(minTsHem, maxTsHem))

# Equatorial Zone
#minTequat = np.min(equat)
#maxTequat = np.max(equat)
#print("Equatorial Zone Temperature Range: {0} to {1}".format(minTequat, maxTequat))

# Set Axis Parameters
# I used the temperature ranges calculated from the max and min values, wanted a consistent y-axis scale and rounded so they would be integers
ymin = -1
ymax = 2

# Set Axis Ranges
ax1.set_ylim(ymin, ymax)
ax2.set_ylim(ymin, ymax)
ax3.set_ylim(ymin, ymax)
ax4.set_ylim(ymin, ymax)

# Titles and Axis Labels

# Axis 1 (Global)
ax1.set_title('Annual Mean Land-Ocean Temperature Index from 1880 to 2024\n(Baseline 1951-1980)\n\nGlobal Mean Temperatures')
ax1.set_ylabel('Temperature ($^\circ$C)')

# Axis 2-4 (Geographical Zones)
ax2.set_title('Mean Temperatures for Different Geographic Climate Zones')
ax2.set_ylabel('N. Hemisphere\nTemperature ($^\circ$C)')
ax3.set_ylabel('S. Hemisphere\nTemperature ($^\circ$C)')
ax4.set_ylabel('Equatorial Zone\n(24$^\circ$S to 24$^\circ$N)\nTemperature ($^\circ$C)')
# Wanting to show x-axis on the bottom most graph only, helps with readibility and reduce clutter on screen
ax4.set_xlabel('Years') 

#plt.tight_layout() #Minimal margins

plt.show()


# OUPUT CSV USING NUMPY

HEADER = "\nTitle: Annual Mean Land-Ocean Temperature Index in degrees Celsius from 1880 to 2024 Compared to Baseline 1951-1980\nSource: NASA GISS, GHCN-v4 1880-04/2024 + SST: ERSST v5 1880-04/2024\nData Methods: Using elimination of outliers and homogeneity adjustment\nMissing Data: Indicated with '*****'\nColumns: 'Year', 'Glob', 'NHem', 'SHem', '25S-24N'\n"

np.savetxt('land_ocean_ladderplot_data.csv', data[:,[0,1,2,3,5]], fmt='%1.2f', delimiter=',', header=HEADER, comments='#')
'''

