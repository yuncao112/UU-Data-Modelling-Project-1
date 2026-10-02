# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 11:38:51 2026

@author: 12619
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# set the project folder
PROJECT_DIR = Path("D:/DaMod-computer-labs/week-02-data-wrangling")

# define paths to subfolders
RAW_DIR = PROJECT_DIR / "data" / "raw"
DAILY_DIR = RAW_DIR / "daily"
ATTRIBUTE_DIR = RAW_DIR / "attributes"
SPATIAL_DIR = RAW_DIR / "spatial"

# list the folders and files
project_items = sorted(PROJECT_DIR.iterdir())
raw_items = sorted(RAW_DIR.iterdir())
daily_files = sorted(DAILY_DIR.glob("*.csv"))

# inspect them
print(project_items)
print(raw_items)
print(len(daily_files))
print(daily_files[:5])

# read the daily data for Thames at Kingston
thames_daily_fn = DAILY_DIR / "camels_gb_v2_hydromet_daily_timeseries_39001_19701001-20220930.csv"

daily = pd.read_csv(thames_daily_fn)

# convert date to datetime
daily["date"] = pd.to_datetime(daily["date"])
# create a new column for year
daily["year"] = daily["date"].dt.year

# count how many days are available in each year
days_per_year = daily.groupby("year").size()

# keep only complete calendar years
complete_years = days_per_year[days_per_year >= 365].index

print(days_per_year.head())
print(days_per_year.tail())
print(complete_years)

# calculate mean annual temperature for each complete year
annual_temperature = (
    daily.loc[daily["year"].isin(complete_years)]
    .groupby("year")["temperature_haduk"]
    .mean()
    .reset_index(name="temperature_mean")
)

print(annual_temperature.head())
print(annual_temperature.tail())

fig, ax = plt.subplots(figsize=(9, 3))

ax.plot(
    annual_temperature["year"],
    annual_temperature["temperature_mean"]
)

ax.set_xlabel("Year")
ax.set_ylabel("Mean annual temperature (deg C)")
ax.set_title("Thames at Kingston mean annual temperature")

plt.show()

# calculate the reference mean temperature
reference_mean_temperature = annual_temperature["temperature_mean"].mean()

# calculate temperature anomaly for each year
annual_temperature["temperature_anomaly"] = (
    annual_temperature["temperature_mean"]
    - reference_mean_temperature
)

print(reference_mean_temperature)
print(annual_temperature.head())
print(annual_temperature.tail())

# calculate the maximum absolute anomaly for colour normalisation
max_abs_anomaly = annual_temperature["temperature_anomaly"].abs().max()

norm = TwoSlopeNorm(
    vmin=-max_abs_anomaly,
    vcenter=0,
    vmax=max_abs_anomaly
)

colors = plt.cm.RdBu_r(
    norm(annual_temperature["temperature_anomaly"])
)

# create climate stripes
fig, ax = plt.subplots(figsize=(9, 3))

ax.bar(
    x=annual_temperature["year"],
    height=1.0,
    color=colors,
    width=1.0
)

# remove axis lines and y-axis labels
for spine in ax.spines.values():
    spine.set_visible(False)

ax.tick_params(
    axis="y",
    left=False,
    labelleft=False
)

ax.set_title(
    "Thames at Kingston climate stripes (temperature anomalies)"
)

plt.show()

# calculate annual total precipitation for complete years
annual_precipitation = (
    daily.loc[daily["year"].isin(complete_years)]
    .groupby("year")["precipitation_haduk"]
    .sum()
    .reset_index(name="precipitation_sum")
)

# calculate reference mean annual precipitation
reference_mean_precipitation = annual_precipitation["precipitation_sum"].mean()

# calculate precipitation anomaly
annual_precipitation["precipitation_anomaly"] = (
    annual_precipitation["precipitation_sum"]
    - reference_mean_precipitation
)

print(reference_mean_precipitation)
print(annual_precipitation.head())
print(annual_precipitation.tail())

max_abs_precip_anomaly = (
    annual_precipitation["precipitation_anomaly"]
    .abs()
    .max()
)

norm = TwoSlopeNorm(
    vmin=-max_abs_precip_anomaly,
    vcenter=0,
    vmax=max_abs_precip_anomaly
)

colors = plt.cm.RdBu_r(
    norm(annual_precipitation["precipitation_anomaly"])
)

fig, ax = plt.subplots(figsize=(9, 3))

ax.bar(
    x=annual_precipitation["year"],
    height=1.0,
    color=colors,
    width=1.0
)

for spine in ax.spines.values():
    spine.set_visible(False)

ax.tick_params(
    axis="y",
    left=False,
    labelleft=False
)

ax.set_title(
    "Thames at Kingston climate stripes (precipitation anomalies)"
)

plt.show()