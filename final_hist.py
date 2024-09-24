# IMPORT MODULES (in ABC order)
import cartopy
import cartopy.crs as ccrs
import glob
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# READING IN THE FILE(S)
os.chdir('csv_data')

# Title: Annual Suicide Mortality (2000-2019) -- ORIGINAL
data_original = pd.read_csv('suicide_mortality_original.csv', delim_whitespace=False, skiprows=0)

# SAVE DATA FROM COUNTRIES
years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]

ss_years = {}

for year in years:
    df_year = data_original[(data_original["DIM_TIME"] == year) & (data_original["DIM_GEO_CODE_TYPE"] == "COUNTRY") & (data_original["DIM_SEX"] == "TOTAL")]
    #print(df_year)

    s_year = pd.Series(data=df_year["RATE_PER_100000_N"].to_list(), index=df_year["DIM_TIME"].to_list())
    #print(s_year)
    ss_years[year] = s_year
    
     # HISTOGRAM
     
    fig = plt.figure(figsize=[10,16])
    ax1 = plt.subplot2grid((9,1), (0,0), rowspan=9)
    rate = s_year
    b = 20
        
    plt.hist(rate, bins=b)
    #ax1.set_ylim(0, 25)
    #ax1.set_ylim(0, 25)
    #plt.set_ylim(bottom=0)  # Adjusting so that the graph is aligned and not "floating"
    #plt.set_xlim(left=0)    # Adjusting so that the graph is aligned and not "floating"
        
    # DYNAMIC PLOT LABELS
    plt.title('Year: {}'.format(year), fontsize=15, ha='center')  # Will change the title based on date
        
    # STATIC FIGURE LABELS
    # Figure Title
    fig.suptitle('Measuring the Frequency of Suicide Mortality Rates\nIn Over 180 Countries and Territories Around the World (2000-2019)', fontsize=16, ha='center')
    # Footer
    fig.text(0.5, 0.025, "Source: World Health Organization 2024 data.who.int, Suicide mortality rate (per 100 000 population) [Indicator]. https://data.who.int/indicators/i/F08B4FD/16BBF41 (Accessed on 1 July 2024)\nVisualized by Lily Gates for AOSC247 at the University of Maryland", fontsize=7, ha='center')
        
    # SET AXIS LIMITS
    plt.xlim([0, 100])
    plt.ylim([0, 70]) 
        
    # STATIC PLOT LABELS
    plt.ylabel('Frequency (Number of Countries/Territories)\n')
    plt.xlabel('\nSuicide Rate (per 100,000 people)\n')  
    
    plt.xticks(np.arange(0, 105, step=5))
    xticks = np.arange(0, 105, 5)
    ax1.set_xticks(xticks, labels=xticks)
    
    plt.yticks(np.arange(0, 75, step=5))
    yticks = np.arange(0, 75, 5)
    ax1.set_yticks(yticks, labels=yticks)

    # Show Figure    
    #plt.show()
    
    # Save Figure
    fig.savefig(f"../hist_plots/plot_{year}")
