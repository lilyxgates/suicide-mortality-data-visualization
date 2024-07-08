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
    print(f"{sex}:")
    print(s_sex)
    ss_sexes[sex] = [s_sex.index.to_numpy(), s_sex.values]


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
ax1.set_xlim(2000, 2020)
ax2.set_xlim(2000, 2020)
ax3.set_xlim(2000, 2020)
ax4.set_xlim(2000, 2020)
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


# Plotting Sexes Data
# SEXES: Total, All Sexes
ax1.scatter(x=ss_sexes["TOTAL"][0], y=ss_sexes["TOTAL"][1], c='k', alpha=0.3, label="Total")
ax1.scatter(x=ss_sexes["FEMALE"][0], y=ss_sexes["FEMALE"][1], c='crimson', alpha=0.3, label="Female")
ax1.scatter(x=ss_sexes["MALE"][0], y=ss_sexes["MALE"][1], c='blue', alpha=0.3, label="Male")

# Individual
ax2.scatter(x=ss_sexes["TOTAL"][0], y=ss_sexes["TOTAL"][1], c='k', alpha=0.3, label="Total")
ax3.scatter(x=ss_sexes["FEMALE"][0], y=ss_sexes["FEMALE"][1], c='crimson', alpha=0.3, label="Female")
ax4.scatter(x=ss_sexes["MALE"][0], y=ss_sexes["MALE"][1], c='blue', alpha=0.3, label="Male")

# Legend
#ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 0.2), fancybox=True, shadow=True, ncol=7)
ax1.legend(loc=0)

# Axis 1 (Global)
ax1.set_title('Annual Suicide Mortality Rates by Sex')
ax1.set_ylabel('Suicide Rate\nper 100,000 people')
ax1.set_xlabel('Years') 

# Axis 2-4 ('Suicide Rate\n(per 100,000 people)')
ax2.set_title('\nLinear Regression for Suicide Mortality Rates')
ax2.set_ylabel('Total')
ax3.set_ylabel('Female')
ax4.set_ylabel('Male')

#Figure Title
fig1.suptitle('Comparing Annual Suicide Mortality Rates (2000-2019)\nBy Sexes', fontsize=16, ha='center')
# Footer
fig1.text(0.5, 0.025, "Source: World Health Organization 2024 data.who.int, Suicide mortality rate (per 100 000 population) [Indicator]. https://data.who.int/indicators/i/F08B4FD/16BBF41 (Accessed on 1 July 2024)\nVisualized by Lily Gates for AOSC247 at the University of Maryland", fontsize=7, ha='center')




plt.show()

######################################
################ AGES ################
######################################

# SAVE AGES DATA
ages = ["TOTAL", "Y15T24", "Y25T34", "Y35T44", "Y45T54", "Y55T64", "Y65T74", "Y75T84", "Y_GE85"]

ss_ages = {}
for age in ages:
    s_age = data_original[(data_original["DIM_AGE"] == age)][["DIM_TIME", "RATE_PER_100000_N"]].groupby(["DIM_TIME"]).mean()
    print(f"{age}:")
    print(s_age)
    ss_ages[age] = [s_age.index.to_numpy(), s_age.values]

# SETTING UP THE FIGURE/PLOTS -- AGES
fig2 = plt.figure(figsize=[16,12])
ax1 = plt.subplot2grid((15,2), (0,0), rowspan=4, colspan=2)  # TOTAL, ALL AGES
#plt.setp(ax2.get_xticklabels(), visible=False) 
ax2 = plt.subplot2grid((15,2), (5,0), rowspan=2)  # Y15T24
ax3 = plt.subplot2grid((15,2), (8,0), rowspan=2)  # Y25T34
ax4 = plt.subplot2grid((15,2), (11,0), rowspan=2)  # Y35T44
ax5 = plt.subplot2grid((15,2), (14,0), rowspan=2)  # Y45T54
ax6 = plt.subplot2grid((15,2), (5,1), rowspan=2)  # Y55T64
ax7 = plt.subplot2grid((15,2), (8,1), rowspan=2)  # Y65T74
ax8 = plt.subplot2grid((15,2), (11,1), rowspan=2)  # Y75T84
ax9 = plt.subplot2grid((15,2), (14,1), rowspan=2)  # Y_GE85



# PLOTTING  DATA
# Setting Axis Limits
#ymin = 5
#ymax = 25

# Finding Max and Min Values for Rates
# Ages
max_total_age = max(ss_ages["TOTAL"][1])
min_total_age = min(ss_ages["TOTAL"][1])

max_15 = max(ss_ages["Y15T24"][1])
min_15 = min(ss_ages["Y15T24"][1])

max_25 = max(ss_ages["Y25T34"][1])
min_25 = min(ss_ages["Y25T34"][1])

max_35 = max(ss_ages["Y35T44"][1])
min_35 = min(ss_ages["Y35T44"][1])

max_45 = max(ss_ages["Y45T54"][1])
min_45 = min(ss_ages["Y45T54"][1])

max_55 = max(ss_ages["Y55T64"][1])
min_55 = min(ss_ages["Y55T64"][1])

max_65 = max(ss_ages["Y65T74"][1])
min_65 = min(ss_ages["Y65T74"][1])

max_75 = max(ss_ages["Y75T84"][1])
min_75 = min(ss_ages["Y75T84"][1])

max_85 = max(ss_ages["Y_GE85"][1])
min_85 = min(ss_ages["Y_GE85"][1])


print(min_total_age)
print(max_total_age)

print(min_15)
print(max_15)

print(min_25)
print(max_25)

print(min_35)
print(max_35)

print(min_45)
print(max_45)

print(min_55)
print(max_55)

print(min_65)
print(max_65)

print(min_75)
print(max_75)

print(min_85)
print(max_85)


# Set Axis Ranges
ax1.set_ylim(0, 32)
ax2.set_ylim(min_15, max_15)
ax3.set_ylim(min_25, max_25)
ax4.set_ylim(min_35, max_35)
ax5.set_ylim(min_45, max_45)
ax6.set_ylim(min_55, max_55)
ax7.set_ylim(min_65, max_65)
ax8.set_ylim(min_75, max_75)
ax9.set_ylim(min_85, max_85)

# AGES: Total, All Ages
ax1.scatter(x=ss_ages["TOTAL"][0], y=ss_ages["TOTAL"][1], c='k', alpha=0.3, label="Total")
ax1.scatter(x=ss_ages["Y15T24"][0], y=ss_ages["Y15T24"][1], c='crimson', alpha=0.3, label="Ages 15-24")
ax1.scatter(x=ss_ages["Y25T34"][0], y=ss_ages["Y25T34"][1], c='peru', alpha=0.3, label="Ages 25-34")
ax1.scatter(x=ss_ages["Y35T44"][0], y=ss_ages["Y35T44"][1], c='goldenrod', alpha=0.3, label="Ages 35-44")
ax1.scatter(x=ss_ages["Y45T54"][0], y=ss_ages["Y45T54"][1], c='yellowgreen', alpha=0.3, label="Ages 45-54")
ax1.scatter(x=ss_ages["Y55T64"][0], y=ss_ages["Y55T64"][1], c='olivedrab', alpha=0.3, label="Ages 55-64")
ax1.scatter(x=ss_ages["Y65T74"][0], y=ss_ages["Y65T74"][1], c='darkcyan', alpha=0.3, label="Ages 65-74")
ax1.scatter(x=ss_ages["Y75T84"][0], y=ss_ages["Y75T84"][1], c='mediumblue', alpha=0.3, label="Ages 75-84")
ax1.scatter(x=ss_ages["Y_GE85"][0], y=ss_ages["Y_GE85"][1], c='purple', alpha=0.3, label="Ages 85+")


# Individual
ax2.scatter(x=ss_ages["Y15T24"][0], y=ss_ages["Y15T24"][1], c='crimson', alpha=0.3, label="Ages 15-24")
ax3.scatter(x=ss_ages["Y25T34"][0], y=ss_ages["Y25T34"][1], c='peru', alpha=0.3, label="Ages 25-34")
ax4.scatter(x=ss_ages["Y35T44"][0], y=ss_ages["Y35T44"][1], c='goldenrod', alpha=0.3, label="Ages 35-44")
ax5.scatter(x=ss_ages["Y45T54"][0], y=ss_ages["Y45T54"][1], c='yellowgreen', alpha=0.3, label="Ages 45-54")
ax6.scatter(x=ss_ages["Y55T64"][0], y=ss_ages["Y55T64"][1], c='olivedrab', alpha=0.3, label="Ages 55-64")
ax7.scatter(x=ss_ages["Y65T74"][0], y=ss_ages["Y65T74"][1], c='darkcyan', alpha=0.3, label="Ages 65-74")
ax8.scatter(x=ss_ages["Y75T84"][0], y=ss_ages["Y75T84"][1], c='mediumblue', alpha=0.3, label="Ages 75-84")
ax9.scatter(x=ss_ages["Y_GE85"][0], y=ss_ages["Y_GE85"][1], c='purple', alpha=0.3, label="Ages 85+")

# TITLES AND AXIS LABELS
#Figure Title
fig2.suptitle('Comparing Annual Suicide Mortality Rates (2000-2019)\nBy Age Group', fontsize=16, ha='center')
# Footer
fig2.text(0.5, 0.025, "Source: World Health Organization 2024 data.who.int, Suicide mortality rate (per 100 000 population) [Indicator]. https://data.who.int/indicators/i/F08B4FD/16BBF41 (Accessed on 1 July 2024)\nVisualized by Lily Gates for AOSC247 at the University of Maryland", fontsize=7, ha='center')

# X-Axis
plt.xticks(np.arange(2000, 2020, step=1))

xticks = np.arange(2000, 2020, 1)
xlabels = ["'00", "'01", "'02", "'03", "'04", "'05", "'06", "'07", "'08", "'09", "'10", "'11", "'12", "'13", "'14", "'15", "'16", "'17", "'18", "'19"]

ax1.set_xticks(xticks, labels=xticks)
ax1.set_xlabel('Years')

ax2.set_xticks(xticks, labels=xlabels)
ax3.set_xticks(xticks, labels=xlabels)
ax4.set_xticks(xticks, labels=xlabels)




plt.show()

"""

#ax1.plot(ss_regions["World"], c='k', linestyle = "--", linewidth = '2', label="Global")  # World
plt.plot(ss_sexes["TOTAL"], c='k', o, label="Total")  # TOTAL

# Plot XY original data
plt.plot(season, snow, 'o', label='Original Data') 

# Running mean 5 year
## Generate Running Mean
# Using the Prof. function
rmsnow = rm.running_mean(snow, 5) 
# Cannot do running mean on first and last 2 values, removes the 0 tales
plt.plot(season[2:-2], rmsnow[2:-2], label='5-yr Running Mean')


"""
