#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 13:40:43 2026

@author: Oski
"""

from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#------------------------ directory-----------------------#

PROJECT_DIR   = Path("/Users/Oski/Desktop/DaMod_computer_labs/week-02-data-wrangling")
RAW_DIR       = PROJECT_DIR / "data" / "raw"
DAILY_DIR     = RAW_DIR / "daily"
ATTRIBUTE_DIR = RAW_DIR / "attributes"
SPATIAL_DIR   = RAW_DIR / "spatial"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
FIGURE_DIR    = PROJECT_DIR / "figures"

#-------------------Settings---------------------------------#
PRECIP_COL                 = "precipitation_haduk"
PET_COL                    = "pet_hydrope"
TEMP_COL                   = "temperature_haduk"
QSPEC_COL                  = "discharge_spec"

#---------------------Time settings---------------------------#

ANALYSIS_START = pd.Timestamp("1991-01-01")
ANALYSIS_END = pd.Timestamp("2020-12-31")

#----------------------Value settings-------------------------#

MIN_VALID_PERCENT_PER_YEAR = 95
MIN_SUFFICIENT_YEARS       = 28

#------------------------Setting up the data frame for gauge 39001

ex_file= DAILY_DIR / "camels_gb_v2_hydromet_daily_timeseries_39001_19701001-20220930.csv"

one_daily = pd.read_csv(ex_file,
                     parse_dates=["date"], 
                     usecols=["date", PRECIP_COL,PET_COL,TEMP_COL,QSPEC_COL])

in_period = (one_daily["date"]>= ANALYSIS_START)&(one_daily["date"]<= ANALYSIS_END)
one_daily= one_daily.loc[in_period]

one_daily["year"] = one_daily["date"].dt.year

#----------------------

one_daily["missing"] = one_daily.isna().any(axis =1 )

an_complete = one_daily.groupby("year").mean()
an_complete["missing"] = an_complete["missing"] * 100 

an_complete["valid_year"] = an_complete["missing"] <= (100- MIN_VALID_PERCENT_PER_YEAR)
num_of_valid_years = an_complete["valid_year"].sum()

#-------------

hydro_attributes = pd.read_csv(ATTRIBUTE_DIR/ "camels_gb_v2_hydrometry_attributes.csv",
                               dtype={"gauge_id":str},
                               usecols = ["gauge_id", "station_quality_qmed"])

#------------------------------

def summmaries_daily_completeness(file_path):
    gauge_id = file_path.name.split("timeseries_")[1].split("_19701001")[0]

    one_daily = pd.read_csv(ex_file,
                         parse_dates=["date"], 
                         usecols=["date", PRECIP_COL,PET_COL,TEMP_COL,QSPEC_COL])

    in_period = (one_daily["date"]>= ANALYSIS_START)&(one_daily["date"]<= ANALYSIS_END)
    one_daily= one_daily.loc[in_period]

    one_daily["year"] = one_daily["date"].dt.year
    
    one_daily["missing"] = one_daily.isna().any(axis =1 )

    an_complete = one_daily.groupby("year").mean()
    an_complete["missing"] = an_complete["missing"] * 100 

    an_complete["valid_year"] = an_complete["missing"] <= (100- MIN_VALID_PERCENT_PER_YEAR)
    num_of_valid_years = an_complete["valid_year"].sum()
    
    row = {
        "gauge_id": gauge_id,
        "num_of_valid_years": num_of_valid_years,
    }

    return row



daily_files = DAILY_DIR.glob("*.csv")


completeness_rows = []
for file_path in daily_files:
    completeness_rows.append(summmaries_daily_completeness(file_path))


daily_completeness = pd.DataFrame(completeness_rows)


station_quality = daily_completeness.merge(hydro_attributes, on="gauge_id", how="left", validate="one_to_one")

#adding columns to station quality 

station_quality["retained_for_analysis"] = station_quality["station_quality_qmed"] & station_quality["num_of_valid_years"].ge(MIN_SUFFICIENT_YEARS)
station_quality.to_csv(PROCESSED_DIR/"camels_gb_1991_2020_station_completeness.csv", index =False)

#-------------------

gb_outline = gpd.read_file(SPATIAL_DIR / "great_britain_outline.gpkg")

# read the topographic attributes and select the required columns
topo_attributes = pd.read_csv(
    ATTRIBUTE_DIR / "camels_gb_v2_topographic_attributes.csv",
    dtype={"gauge_id": str},
    usecols=["gauge_id", "gauge_easting", "gauge_northing"]
)

# merge to station_quality to add the spatial information
station_quality_spat = station_quality.merge(topo_attributes, on="gauge_id", how="left").copy()

# filter the table for retained and excluded stations
retained_stations = station_quality_spat.loc[station_quality_spat["retained_for_analysis"]]
excluded_stations = station_quality_spat.loc[~station_quality_spat["retained_for_analysis"]]   

# make the map
fig, ax = plt.subplots(figsize=(6, 6))
gb_outline.plot(ax=ax, color="lightgrey", linewidth=1)
ax.scatter(
    retained_stations["gauge_easting"],
    retained_stations["gauge_northing"],
    c="darkblue",
    s=10,
    label="Retained"
)
ax.scatter(
    excluded_stations["gauge_easting"],
    excluded_stations["gauge_northing"],
    c="red",
    s=10,
    label="Excluded"
)

ax.set_title("Retained vs excluded stations")
ax.set_axis_off()
ax.legend(loc="upper right")


#------------------------------

#-------------------------------Derive statistics----------------------------#

#Setting up the function 

def derive_stats(file_path):
    #Reading the Gauge id from each daily file in file path 
    gauge_id = file_path.name.split("timeseries_")[1].split("_19701001")[0]
    
    one_daily=pd.read_csv(file_path,
                          parse_dates=["date"],
                          usecols=["date", PRECIP_COL,PET_COL,TEMP_COL,QSPEC_COL]
               )

    one_daily = one_daily.loc[(one_daily["date"]>= ANALYSIS_START) & (one_daily["date"]<=ANALYSIS_END)]

#Calculating the Mean 
    p_mean = one_daily[PRECIP_COL].mean()
    pet_mean = one_daily[PET_COL].mean()
    t_mean = one_daily[TEMP_COL].mean()
    q_mean = one_daily[QSPEC_COL].mean()

#Calculating the aridity index 

    aridity = pet_mean/p_mean

#Calculating the precip fraction on days colder than 0C

    total_precip = one_daily[PRECIP_COL].sum()
    total_snowfall = one_daily[one_daily[TEMP_COL]< 0][PRECIP_COL].sum()
    snow_frac = total_snowfall/total_precip if total_precip != 0 else 0

#Calculating high precipitation frequency 

    high_prec_freq =one_daily[one_daily[PRECIP_COL]>= 5*p_mean].shape[0]

#Calculating the 5% and 95% daily flow quantiles 

    Q5 = one_daily[QSPEC_COL].quantile(0.05)

    Q95 = one_daily[QSPEC_COL].quantile(0.95)

#Calculating the Runoff ratio 
    runoff_ratio = q_mean/p_mean

    return { 
       "gauge_id":gauge_id,
       "p_mean": p_mean,
       "pet_mean": pet_mean, 
       "t_mean":t_mean,
       "q_mean": q_mean, 
       "aridity": aridity,
       "snow_frac":snow_frac, 
       "high_prec_freq": high_prec_freq,
       "Q5":Q5,
       "Q95":Q95,
       "runoff_ratio":runoff_ratio,
       }

#producing a list of retained gauges 
retained_gauges = station_quality[station_quality["retained_for_analysis"] == True]["gauge_id"]
retained_files = [DAILY_DIR / f"camels_gb_v2_hydromet_daily_timeseries_{gauge_id}_19701001-20220930.csv" for gauge_id in retained_gauges]

#creating a loop applying the function across all files in the retained files

derived_rows = []
for file_path in retained_files:
    derived_rows.append(derive_stats(file_path))

derived_stat_table = pd.DataFrame(derived_rows)

#inspecting derived stats 
derived_stat_table.head()

derived_stat_table.describe()

#------------ Merging catchment attributes------------------#

#defining columns 
soil_columns = [
    "gauge_id",
    "sand_perc",
    "silt_perc",
    "clay_perc",
    "organic_perc",
    "bulkdens",
    "tawc",
    "porosity_hypres",
    "conductivity_hypres",
    "root_depth",
    "soil_depth_pelletier",
]
landcover_columns = [
    "gauge_id",
    "dwood_perc_2015",
    "ewood_perc_2015",
    "grass_perc_2015",
    "crop_perc_2015",
    "urban_perc_2015",
]

topography = pd.read_csv(ATTRIBUTE_DIR/ "camels_gb_v2_topographic_attributes.csv", 
                         dtype={"gauge_id":str}) 
soils =  pd.read_csv(ATTRIBUTE_DIR/ "camels_gb_v2_soil_attributes.csv",
                     dtype={"gauge_id":str},
                     usecols = soil_columns)
land_cover = pd.read_csv(ATTRIBUTE_DIR/ "camels_gb_v2_landcover_attributes.csv",
                         dtype={"gauge_id":str},
                         usecols=landcover_columns)
#merging the dervied stats with the soil attribute data 
final_table = (
    derived_stat_table
    .merge(topography, on="gauge_id", how="left", validate="one_to_one")
    .merge(soils, on="gauge_id", how="left", validate="one_to_one")
    .merge(land_cover, on="gauge_id", how="left", validate="one_to_one")
)

final_table_path = PROCESSED_DIR / "camels_gb_1991_2020_analysis_ready.csv"
final_table.to_csv(final_table_path, index=False)

#-------------------Data analysis------------------------#

#Exploring data before modelling 

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.hist(final_table["runoff_ratio"], bins=30, color="tab:blue", edgecolor="white")
ax.set(xlabel="Runoff ratio", ylabel="Number of catchments", title="Distribution of 1991-2020 runoff ratio")
plt.show()



fig_aridity, ax = plt.subplots(figsize=(5, 4))
ax.scatter(final_table["aridity"], final_table["runoff_ratio"], s=25, alpha=0.3, color="tab:blue")
ax.set(xlabel="Aridity (PET / P)", ylabel="Runoff ratio")
plt.title("Correlation between Aridity and Runoff")
plt.show()

fig_aridity, ax = plt.subplots(figsize=(5, 4))
ax.scatter(final_table["urban_perc_2015"], final_table["runoff_ratio"], s=25, alpha=0.3, color="tab:blue")
ax.set(xlabel="Urban_cover_%", ylabel="Runoff ratio")
plt.title("Correlation between Urban land cover and Runoff")
plt.show()



#----Plotting maps


# read the catchments outlines and the Great Britain outline
catchments = gpd.read_file(SPATIAL_DIR / "camels_gb_v2_catchments.gpkg")
gb_outline = gpd.read_file(SPATIAL_DIR / "great_britain_outline.gpkg")

# merge the catchments with the final table to get a GeoDataFrame with all the attributes that we prepared
catchment_map = catchments.merge(final_table, left_on="ID_STRING", right_on="gauge_id", how="right", validate="one_to_one")

fig, ax = plt.subplots(figsize=(4, 6))
gb_outline.boundary.plot(ax=ax, color="0.4", linewidth=0.6)
catchment_map.plot(
    column="runoff_ratio",
    ax=ax,
    legend=True,
    cmap="viridis",
    legend_kwds={
        "label": "Runoff ratio",
        "shrink": 0.40,
        "aspect": 20,
    }
)
ax.set_axis_off()
plt.title("Spatial distribution of runoff across the UK")
plt.show()

fig, ax = plt.subplots(figsize=(4, 6))
gb_outline.boundary.plot(ax=ax, color="0.4", linewidth=0.6)
catchment_map.plot(
    column="aridity",
    ax=ax,
    legend=True,
    cmap="viridis",
    legend_kwds={
        "label": "Aridity ",
        "shrink": 0.40,
        "aspect": 20,
    }
)
ax.set_axis_off()
plt.title("Spatial distribution of aridity across the UK")
plt.show()

fig, ax = plt.subplots(figsize=(4, 6))
gb_outline.boundary.plot(ax=ax, color="0.4", linewidth=0.6)
catchment_map.plot(
    column="root_depth",
    ax=ax,
    legend=True,
    cmap="viridis",
    legend_kwds={
        "label": "root_depth",
        "shrink": 0.40,
        "aspect": 20,
    }
)
ax.set_axis_off()
plt.title("Spatial distribution of root depth (m) across the UK")
plt.show()

fig, ax = plt.subplots(figsize=(4, 6))
gb_outline.boundary.plot(ax=ax, color="0.4", linewidth=0.6)
catchment_map.plot(
    column="bulkdens",
    ax=ax,
    legend=True,
    cmap="viridis",
    legend_kwds={
        "label": "Bulk density (g cm-3)",
        "shrink": 0.40,
        "aspect": 20,
    }
)
ax.set_axis_off()
plt.title("Spatial distribution of bulk density across the UK")
plt.show()

ig, ax = plt.subplots(figsize=(4, 6))
gb_outline.boundary.plot(ax=ax, color="0.4", linewidth=0.6)
catchment_map.plot(
    column="porosity_hypres",
    ax=ax,
    legend=True,
    cmap="viridis",
    legend_kwds={
        "label": "Soil porosity",
        "shrink": 0.40,
        "aspect": 20,
    }
)
ax.set_axis_off()
plt.title("Spatial distribution of Soil porosity across the UK")
plt.show()

fig, ax = plt.subplots(figsize=(4, 6))
gb_outline.boundary.plot(ax=ax, color="0.4", linewidth=0.6)
catchment_map.plot(
    column="soil_depth_pelletier",
    ax=ax,
    legend=True,
    cmap="viridis",
    legend_kwds={
        "label": "Soil depth(m)",
        "shrink": 0.40,
        "aspect": 20,
    }
)
ax.set_axis_off()
plt.title("Spatial distribution of Soil depth (m) across the UK")
plt.show()

fig, ax = plt.subplots(figsize=(4, 6))
gb_outline.boundary.plot(ax=ax, color="0.4", linewidth=0.6)
catchment_map.plot(
    column="urban_perc_2015",
    ax=ax,
    legend=True,
    cmap="viridis",
    legend_kwds={
        "label": "Urban land cover (%)",
        "shrink": 0.40,
        "aspect": 20,
    }
)
ax.set_axis_off()
plt.title("Spatial distribution of Urban land cover fraction across the UK")
plt.show()

fig, ax = plt.subplots(figsize=(4, 6))
gb_outline.boundary.plot(ax=ax, color="0.4", linewidth=0.6)
catchment_map.plot(
    column="Q5",
    ax=ax,
    legend=True,
    cmap="viridis",
    legend_kwds={
        "label": "5% Discharge quantile",
        "shrink": 0.40,
        "aspect": 20,
    }
)
ax.set_axis_off()
plt.title("Spatial distribution of Q5 discharge across the UK")
plt.show()

fig, ax = plt.subplots(figsize=(4, 6))
gb_outline.boundary.plot(ax=ax, color="0.4", linewidth=0.6)
catchment_map.plot(
    column="Q95",
    ax=ax,
    legend=True,
    cmap="viridis",
    legend_kwds={
        "label": "95% Discharge Quantile",
        "shrink": 0.40,
        "aspect": 20,
    }
)
ax.set_axis_off()
plt.title("Spatial distribution of Q95 across the UK")
plt.show()



