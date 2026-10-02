#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 12:18:34 2026

@author: Oski
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

#-----------------------Setting  up environment and directory

# set the project folder to the folder where you unpacked the zip archive
PROJECT_DIR   = Path("/Users/Oski/Desktop/DaMod_computer_labs/week-02-data-wrangling")

# define paths to subfolders
RAW_DIR       = PROJECT_DIR / "data" / "raw"
DAILY_DIR     = RAW_DIR / "daily"
ATTRIBUTE_DIR = RAW_DIR / "attributes"
SPATIAL_DIR   = RAW_DIR / "spatial"



project_items = sorted(PROJECT_DIR.iterdir())
raw_items     = sorted(RAW_DIR.iterdir())
daily_files   = sorted(DAILY_DIR.glob("*.csv"))

#daily files description and overview 
project_items
raw_items
len(daily_files)
daily_files[:5]


#------------------------------------data set up------------------------#

#Reading data in python 
thames_daily_fn = DAILY_DIR / "camels_gb_v2_hydromet_daily_timeseries_39001_19701001-20220930.csv"
daily           = pd.read_csv(thames_daily_fn)

#investigating different aspects of the data 
daily.head()
daily.shape
daily.dtypes
daily.describe()

#converting date column to a datetime column 
daily["date"]=pd.to_datetime(daily["date"])

date_start = daily["date"].min()
date_end= daily["date"].max()
date_start, date_end

#investigating missing data 
missing_count = daily.isna().sum()
missing_percent = daily.isna().mean()*100
missing_table = pd.DataFrame({"missing_days": missing_count,
                              "missing_percent": missing_percent.round(2)})

print(missing_table)

#----------------------Exploring spatial data-------------------#

topo_attri = pd.read_csv(ATTRIBUTE_DIR/ "camels_gb_v2_topographic_attributes.csv", 
                         dtype={"gauge_id":str})
topo_attri.head()

#preparing Geodataframe using topographical data needed for station maps 
station_col = ["gauge_id", "gauge_name", "gauge_easting", "gauge_northing"]
stations = pd.read_csv(ATTRIBUTE_DIR/"camels_gb_v2_topographic_attributes.csv",
                       dtype={"gauge_id": str}, 
                       usecols = station_col)
station_points = gpd.GeoDataFrame(
    stations,
    geometry=gpd.points_from_xy(stations["gauge_easting"], stations["gauge_northing"]),
    crs="EPSG:27700",
)

station_points.head()

gb_outline = gpd.read_file(SPATIAL_DIR / "great_britain_outline.gpkg")
catchments = gpd.read_file(SPATIAL_DIR / "camels_gb_v2_catchments.gpkg" )

gb_outline.crs
catchments.crs
catchments.head()

station_points.crs

# plotting map of gauging stations 

fig, ax = plt.subplots(figsize= (5,7))
gb_outline.boundary.plot(ax=ax, color = "0.4", linewidth= 0.8)
station_points.plot(ax=ax, markersize = 8, color = "tab:blue", alpha = 0.7)
ax.set_title("CAMELS-GB v2 gauging stations")
ax.set_axis_off()
plt.show()

# plotting map of catchment outlines 

fig,ax = plt.subplots(figsize= (5,7))
gb_outline.boundary.plot(ax=ax, color = "0.4", linewidth = 0.8)
catchments.plot(ax = ax, color = "tab:green", linewidth = 0.5)
ax.set_title("CAMELS-GB v2 catchment boundaries")
ax.set_axis_off()
plt.show()

plt.close("all")

station_points.crs

#------------------------------Climate stripes----------------------_# 
daily["year"]= daily["date"].dt.year

days_per_yr= daily.groupby("year").size()
complete_yr= days_per_yr[days_per_yr >= 365].index

annual_temp = (daily.loc[daily["year"].isin(complete_yr)]
               .groupby("year")["temperature_haduk"]
               .mean()
               .reset_index(name = "temperature mean"))

fig, ax = plt.subplots(figsize=(9, 3))
ax.plot(annual_temp["year"], annual_temp["temperature mean"], color="tab:blue")
ax.set_xlabel("Year")
ax.set_ylabel("Mean annual temperature (deg C)")
ax.set_title("Thames at Kingston mean annual temperature")
plt.show()


annual_temp.head()
annual_temp.tail()

ref_mean_temp = annual_temp["temperature mean"].mean()

annual_temp["temperature_anomaly"]= annual_temp["temperature mean"]-ref_mean_temp
annual_temp.tail()

max_abs_anorm = annual_temp["temperature_anomaly"].abs().max()
norm = TwoSlopeNorm(vmin = -max_abs_anorm, vcenter= 0, vmax=max_abs_anorm)
colors=plt.cm.RdBu_r(norm(annual_temp["temperature_anomaly"]))

fig, ax= plt.subplots(figsize=(9,3))
ax.bar(
       x = annual_temp["year"], 
       height = 1.0, 
       color = colors, 
       width = 1.0)

for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(
    axis = "y", 
    left = False,
    labelleft = False)

ax.set_title("Thames at Kingston climate stripes (temperature anomalies)")

plt.show()

#----------------------------------Climate stripes for preciptation-----------#

 
plt.close("all")

annual_precip = (daily.loc[daily["year"].isin(complete_yr)]
.groupby("year")["precipitation_haduk"]
.sum()
.reset_index(name = "precipitation_sum"))

annual_precip.head()
annual_precip.tail()

ref_mean_precip = annual_precip["precipitation_sum"].mean()
annual_precip["precipitation_anomaly" ] = annual_precip["precipitation_sum"]-ref_mean_precip

max_abs_anorm_precip = annual_precip["precipitation_anomaly"].abs().max()
norm_precip = TwoSlopeNorm(vmin = -max_abs_anorm_precip, vcenter= 0, vmax=max_abs_anorm_precip)
colors=plt.cm.RdBu_r(norm_precip(annual_precip["precipitation_anomaly"]))

fig, ax= plt.subplots(figsize=(9,3))
ax.bar(
       x = annual_precip["year"], 
       height = 1.0, 
       color = colors, 
       width = 1.0)

for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(
    axis = "y", 
    left = False,
    labelleft = False)

ax.set_title("Thames at Kingston climate stripes (precipitation anomalies)")

plt.show()
