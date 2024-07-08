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
data_original = pd.read_csv('suicide_mortality_original.csv', delim_whitespace=False, header=None, skiprows=1).values
#print("ORIGINAL")
#print(data_original)

'''
# Title: Annual Suicide Mortality (2000-2019) -- GLOBAL
data_world = pd.read_csv('suicide_mortality_global.csv', delim_whitespace=False, header=None, skiprows=2).values
#print("WORLD")
#print(data_global)

# Title: Annual Suicide Mortality (2000-2019) -- REGIONS
data_regions = pd.read_csv('suicide_mortality_regions.csv', delim_whitespace=False, header=None, skiprows=2).values
#print("REGIONS")
#print(data_regions)

# Title: Annual Suicide Mortality (2000-2019) -- COUNTRIES
data_countries = pd.read_csv('suicide_mortality_countries.csv', delim_whitespace=False, header=None, skiprows=2).values
#print("COUNTRIES")
#print(data_countries)

# Title: Annual Suicide Mortality (2000-2019) -- INCOMES
data_incomes = pd.read_csv('suicide_mortality_incomes.csv', delim_whitespace=False, header=None, skiprows=2).values
#print("INCOMES")
#print(data_incomes)
'''

### 4 Year
### 7 Region -- (Country, WHO Region, Global, World Bank Income Group)
### 10 Geographic Name -- (Country Names, Regions[6] (Africa, Americas, Eastern Mediterranean, Europe, South-East Asia, Western Pacific), Global (World), Income[4] (Low, Lower-Middle, Upper-Middle, High))
### 11 Sex[3] -- (Total, Male Female)
### 12 Age[9] -- (Total, 15-24, 25-34, 35-44, 45-54, 55-64, 65-74, 75-84, 85+)
### 13 Rate per 100,000
### 14 Rate per 100,000 -- (Lower Limit)
### 15 Rate per 100,000 -- (Upper Limit)

# Masking "nan" Values
masked_data = np.ma.masked_where(data_original == "nan", data_original)
clean_data = np.ma.compress_rows(masked_data)

# ASSIGN VARIABLES
# Starting at row 1 because row 0 is the header names
years = clean_data[:, 4]   # x-axis for all graphs
qual = clean_data[:, 10]
#world = clean_data[:, 10]
#countries = clean_data[:,10]
#income = clean_data[:,10]
sexes = clean_data[:, 11]
ages = clean_data[:, 12]
rates = clean_data[:, 13]


# SAVE REGIONAL DATA

# AFRICA (Masking all values except Africa)
masked_africa = np.ma.masked_where(qual != "Africa", clean_data)
clean_africa = np.ma.compress_rows(masked_africa)

#print(masked_africa)
print(clean_africa)

# AMERICAS
# EASTERN MEDITERRANEAN
# EUROPE
# SOUTH-EAST ASIA
# WESTERN PACIFIC




#print(years)
#print(world)
#print(regions)
#print(countries)
#print(income)
#print(ages)




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

