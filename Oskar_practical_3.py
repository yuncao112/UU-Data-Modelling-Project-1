#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 11:16:01 2026

@author: Oski
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


PROJECT_DIR   = Path("/Users/Oski/Desktop/DaMod_computer_labs/week-02-data-wrangling")
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
FIGURE_DIR    = PROJECT_DIR / "figures"

final_table_path = PROCESSED_DIR / "camels_gb_1991_2020_analysis_ready.csv"

# Loading the processed table from pracitcal 2 
final_table = pd.read_csv(final_table_path)

# Checking the table 
final_table.head()


#---------------------1: stat ference and correlation---------------#
#Revisting run off distribution 
fig, ax = plt.subplots(figsize =(6,3))
ax.hist(final_table["runoff_ratio"], bins= 30, color = "tab:blue", edgecolor = "white")
ax.set_xlabel("Runoff ratio")
ax.set_ylabel("Number of catchments")
ax.set_title("Distribution of runoff ratio")
plt.show()

#
rr= final_table["runoff_ratio"]

mean_rr= rr.mean()
median_rr = rr.median()
sigma_rr = rr.std(ddof=1)

print(f"Mean         : {mean_rr:.3f}")
print(f"Median         : {median_rr:.3f}")
print(f"Standard         : {sigma_rr:.3f}")

fig, ax = plt.subplots(figsize = (6,3.5))
ax.hist(rr, bins = 30, color = "tab:blue",  edgecolor= "white", density = True)
x_rr = np.linspace(rr.min(), rr.max(),200) 
pdf_rr =stats.norm.pdf(x_rr, mean_rr, sigma_rr) 
ax.plot(x_rr, pdf_rr, color="black", linewidth=2, label="Normal distribution")

ax.set_xlabel("Runoff ratio")
ax.set_ylabel("Probability density")
ax.legend()
plt.show()

#Applying the afromentioned analysis for raw q_mean 

q = final_table["q_mean"]


mean_q= q.mean()
median_q = q.median()
sigma_q = q.std(ddof=1)

print(f"Mean         : {mean_q:.3f}")
print(f"Median         : {median_q:.3f}")
print(f"Standard         : {sigma_q:.3f}")

fig, ax = plt.subplots(figsize = (6,3.5))
ax.hist(q, bins = 30, color = "tab:blue",  edgecolor= "white", density = True)
x_q = np.linspace(q.min(), q.max(),200) 
pdf_q =stats.norm.pdf(x_q, loc = mean_q, scale = sigma_q) 
ax.plot(x_q, pdf_q, color="black", linewidth=2, label="Normal distribution")

ax.set_xlabel("Mean discharge (mm d-1)")
ax.set_ylabel("Probability density")
ax.legend()
plt.show()

# applying for logged q 

q_positive =  q[q>0]

log_q = np.log10(q_positive)


log_mean_q= log_q.mean()
log_median_q = log_q.median()
log_sigma_q = log_q.std(ddof=1)

print(f"Mean         : {log_mean_q:.3f}")
print(f"Median         : {log_median_q:.3f}")
print(f"Standard         : {log_sigma_q:.3f}")

fig, ax = plt.subplots(figsize = (6,3.5))
ax.hist(log_q, bins = 30, color = "tab:blue",  edgecolor= "white", density = True)
x_log_q = np.linspace(log_q.min(), log_q.max(),200) 
pdf_log_q =stats.norm.pdf(x_log_q, loc = log_mean_q, scale = log_sigma_q) 
ax.plot(x_log_q, pdf_log_q, color="black", linewidth=2, label="Normal distribution")

ax.set_xlabel("Mean discharge (mm d-1)")
ax.set_ylabel("Probability density")
ax.legend()
plt.show()

#--------------------Q2 sample uncertainty---------------------

n = len(rr)

se_rr = sigma_rr/np.sqrt(n) 

print(f"Number of catchments: {n}")
print(f"Standard error: {se_rr:.4f}")

ci_rr = stats.t.interval(
    confidence = 0.95, 
    df = n -1, 
    loc = mean_rr, 
    scale= se_rr
    )
print(f"95% confidence interval: {ci_rr[0]:.3f} - {ci_rr[1]:.3f}")



fig, ax = plt.subplots(figsize=(6.5, 3.5))
ax.hist(rr, bins=30, color="tab:blue", edgecolor="white",
    density=True, alpha=0.6, label="Observed distribution")

ax.plot(x_rr, pdf_rr, color="black", linewidth=2, label="Normal distribution")


ax.axvline(mean_rr, color="tab:red", lw=2,label="Sample mean")
ax.axvspan(ci_rr[0], ci_rr[1], color="tab:red", alpha=0.2, label="95% confidence interval")

ax.set_xlabel("Runoff ratio")
ax.set_ylabel("Probability density")
ax.set_title("Runoff ratio distribution and uncertainty in the mean")
ax.legend()
plt.show()


n_q  = len(q) 
se_q = sigma_q/np.sqrt(n_q)
ci_q = stats.t.interval(
    confidence = 0.95, 
    df = n -1, 
    loc = mean_q, 
    scale= se_q
    )

n_log = len(log_q)
se_log = log_sigma_q/np.sqrt(n_log)
ci_log = stats.t.interval(
    confidence = 0.95, 
    df = n -1, 
    loc = log_mean_q, 
    scale= se_log,
    )

print(f"Number of catchments: {n_q}")
print(f"Standard error: {se_q:.4f}")

print(f"Number of catchments: {n_log}")
print(f"Standard error: {se_log:.4f}")


fig, ax = plt.subplots(figsize=(6.5, 3.5))
ax.hist(q, bins=30, color="tab:blue", edgecolor="white",
    density=True, alpha=0.6, label="Observed distribution")

ax.plot(x_q, pdf_q, color="black", linewidth=2, label="Normal distribution")


ax.axvline(mean_q, color="tab:red", lw=2,label="Sample mean")
ax.axvspan(ci_q[0], ci_q[1], color="tab:red", alpha=0.2, label="95% confidence interval")

ax.set_xlabel("Discharge mm d-1")
ax.set_ylabel("Probability density")
ax.set_title("Discharge distribution and uncertainty in the mean")
ax.legend()
plt.show()

fig, ax = plt.subplots(figsize=(6.5, 3.5))
ax.hist(log_q, bins=30, color="tab:blue", edgecolor="white",
    density=True, alpha=0.6, label="Observed distribution")

ax.plot(x_log_q, pdf_log_q, color="black", linewidth=2, label="Normal distribution")


ax.axvline(log_mean_q, color="tab:red", lw=2,label="Sample mean")
ax.axvspan(ci_log[0], ci_log[1], color="tab:red", alpha=0.2, label="95% confidence interval")

ax.set_xlabel(" Logged Discharge mm d-1")
ax.set_ylabel("Probability density")
ax.set_title(" Logged Discharge distribution and uncertainty in the mean")
ax.legend()
plt.show()

print(f"95% confidence interval: {ci_q[0]:.3f} - {ci_q[1]:.3f}")
print(f"95% confidence interval: {ci_log[0]:.3f} - {ci_log[1]:.3f}")


#-------------------------Q3 correlation analysis 

arid = final_table["aridity"]
fig, ax = plt.subplots(figsize=(5, 4))

ax.scatter(arid,rr,s=25,alpha=0.3,color="tab:blue")
ax.set_xlabel("Aridity index (PET / P)")
ax.set_ylabel("Runoff ratio")
plt.show()

stats.pearsonr(arid, rr, method= None) 


fig, ax = plt.subplots(figsize=(5, 4))

points = ax.scatter(arid,rr,s=30,alpha=0.3,c =final_table["gauge_lat"] , cmap="managua")

plt.colorbar(points, ax=ax, label = "Latitude")
ax.set_xlabel("Aridity index (PET / P)")
ax.set_ylabel("Runoff ratio")
plt.show()


#correlation between rr and urban cover 

urban_cover = final_table["urban_perc_2015"]
fig, ax = plt.subplots(figsize=(5, 4))

ax.scatter(urban_cover,rr,s=25,alpha=0.3,color="tab:blue")
ax.set_xlabel("Urban and suburban cover%")
ax.set_ylabel("Runoff ratio")
plt.show()

stats.pearsonr(urban_cover, rr, method= None) 


fig, ax = plt.subplots(figsize=(5, 4))

points = ax.scatter(urban_cover,rr,s=30,alpha=0.3,c =final_table["gauge_lat"] , cmap="managua")

plt.colorbar(points, ax=ax, label = "Latitude")
ax.set_xlabel("Urban and suburban cover%")
ax.set_ylabel("Runoff ratio")
plt.show()


#correlation between rr and total water content 

water_content = final_table["tawc"]
fig, ax = plt.subplots(figsize=(5, 4))

ax.scatter(water_content,rr,s=25,alpha=0.3,color="tab:blue")
ax.set_xlabel("Total availble water content(mm)")
ax.set_ylabel("Runoff ratio")
plt.show()

stats.pearsonr(water_content, rr, method= None) 


fig, ax = plt.subplots(figsize=(5, 4))

points = ax.scatter(water_content,rr,s=30,alpha=0.3,c =final_table["gauge_lat"] , cmap="managua")

plt.colorbar(points, ax=ax, label = "Latitude")
ax.set_xlabel("Total availble water content(mm)")
ax.set_ylabel("Runoff ratio")
plt.show()

#correlation between rr and dpsbar

dpsbar = final_table["dpsbar"]
fig, ax = plt.subplots(figsize=(5, 4))

ax.scatter(dpsbar,rr,s=25,alpha=0.3,color="tab:blue")
ax.set_xlabel("Catchment mean drainage path slope (m km-1")
ax.set_ylabel("Runoff ratio")
plt.show()

stats.pearsonr(dpsbar, rr, method= None) 


fig, ax = plt.subplots(figsize=(5, 4))

points = ax.scatter(dpsbar,rr,s=30,alpha=0.3,c =final_table["gauge_lat"] , cmap="managua")

plt.colorbar(points, ax=ax, label = "Latitude")
ax.set_xlabel("Catchment mean drainage path slope (m km-1)")
ax.set_ylabel("Runoff ratio")
plt.show()

stats.pearsonr(dpsbar, rr, method= None)



#---------------------------Question 4-----------------

#fitting regression line 
slope, intercept = np.polyfit(arid,rr, 1)
print(f"Slope = {slope:.3f}")
print(f"Intercept = {intercept:.3f}")

#putting regression line onto graph
rr_predicted = intercept + slope * arid

fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(arid, rr, s=25, alpha=0.4, label="Catchments")
ax.plot(arid, rr_predicted, color="black", lw=2, label="Linear regression")
ax.set_xlabel("Aridity index (PET / P)")
ax.set_ylabel("Runoff ratio")
ax.legend()
plt.show()

#Sum of squares and R2 goodness of fit

ss_res = np.sum((rr-rr_predicted)**2)
ss_tot = np.sum((rr - rr.mean())**2)
r2 = 1 - ss_res/ss_tot
print(r2)


#Analyzing residuals 
residuals = (rr - rr_predicted)

print(residuals.describe())

fig, ax = plt.subplots(figsize=(5, 4))

ax.scatter(arid, residuals, s=25, alpha=0.4)
ax.axhline(0, color="black", lw=1)
ax.set_xlabel("Aridity index (PET / P)")
ax.set_ylabel("Residuals")
plt.show()

fig, ax = plt.subplots(figsize=(6, 3.5))

ax.hist(residuals, bins=30, edgecolor="white")
ax.axvline(0, color="black", lw=1)

ax.set_xlabel("Residuals")
ax.set_ylabel("Number of catchments")
ax.set_title("Distribution of regression residuals")
plt.show()


#Regression slope uncertainty 


se_slope = np.sqrt(np.sum(residuals**2) /(n - 2)/np.sum((arid - arid.mean())**2))

t_value = stats.t.ppf(0.975, n - 2)

slope_lower = slope - t_value * se_slope
slope_upper = slope + t_value * se_slope

print(f"Slope estimate = {slope:.3f}")
print(f"95% confidence interval: {slope_lower:.3f} to {slope_upper:.3f}")

#------------------Preforming my own regressional analysis

#-------Land cover: Forest cover 
# What is the relationship between the dedicous forest cover and runoff ratio in the year 2015 
#Hypothesis: Catchments with higher forest cover will have reduced runoff

# creating a correlation plot 

deciduous_cover = final_table["dwood_perc_2015"]
fig, ax = plt.subplots(figsize=(5, 4))

ax.scatter(deciduous_cover,rr,s=25,alpha=0.3,color="tab:blue")
ax.set_xlabel("Deciduous cover%")
ax.set_ylabel("Runoff ratio")
plt.show()


fig, ax = plt.subplots(figsize=(5, 4))

points = ax.scatter(deciduous_cover,rr,s=30,alpha=0.3,c =final_table["gauge_lat"] , cmap="managua")

plt.colorbar(points, ax=ax, label = "Latitude")
ax.set_xlabel("Deciduous cover%")
ax.set_ylabel("Runoff ratio")
plt.show()


# pearsons correlation rank 
stats.pearsonr(deciduous_cover, rr, method= None) 

# fitting a linear regression 
slope, intercept = np.polyfit(deciduous_cover,rr, 1)

rr_predicted = intercept + slope * deciduous_cover

fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(deciduous_cover, rr, s=25, alpha=0.4, label="Catchments")
ax.plot(deciduous_cover, rr_predicted, color="black", lw=2, label="Linear regression")
ax.set_xlabel("Deciduous cover %")
ax.set_ylabel("Runoff ratio")
ax.legend()
plt.show()

#Assessing quality 

ss_res = np.sum((rr-rr_predicted)**2)
ss_tot = np.sum((rr - rr.mean())**2)
r2 = 1 - ss_res/ss_tot
print(r2)

residuals = (rr - rr_predicted)

print(residuals.describe())

fig, ax = plt.subplots(figsize=(5, 4))

ax.scatter(deciduous_cover, residuals, s=25, alpha=0.4)
ax.axhline(0, color="black", lw=1)
ax.set_xlabel("Deciduous cover %")
ax.set_ylabel("Residuals")
plt.show()



 #--------Soil: root depth 
 
 # What is the relationship between the root depth and runoff ratio across catchments in the UK in 2015 
 #Hypothesis: Catchments with deeper root depth will have reduced runoff

 # creating a correlation plot 

root_d= final_table["root_depth"]
fig, ax = plt.subplots(figsize=(5, 4))

ax.scatter(root_d,rr,s=25,alpha=0.3,color="tab:blue")
ax.set_xlabel("Root depth (m)")
ax.set_ylabel("Runoff ratio")
plt.show()


fig, ax = plt.subplots(figsize=(5, 4))

points = ax.scatter(root_d,rr,s=30,alpha=0.3,c =final_table["gauge_lat"] , cmap="managua")

plt.colorbar(points, ax=ax, label = "Latitude")
ax.set_xlabel("Root depth (m)")
ax.set_ylabel("Runoff ratio")
plt.show()


 # pearsons correlation rank 
stats.pearsonr(root_d, rr, method= None) 

 # fitting a linear regression 
slope, intercept = np.polyfit(root_d,rr, 1)

rr_predicted = intercept + slope * root_d

fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(root_d, rr, s=25, alpha=0.4, label="Catchments")
ax.plot(root_d, rr_predicted, color="black", lw=2, label="Linear regression")
ax.set_xlabel("Root depth (m)")
ax.set_ylabel("Runoff ratio")
ax.legend()
plt.show()

 #Assessing quality 

ss_res = np.sum((rr-rr_predicted)**2)
ss_tot = np.sum((rr - rr.mean())**2)
r2 = 1 - ss_res/ss_tot
print(r2)

residuals = (rr - rr_predicted)

print(residuals.describe())

fig, ax = plt.subplots(figsize=(5, 4))

ax.scatter(root_d, residuals, s=25, alpha=0.4)
ax.axhline(0, color="black", lw=1)
ax.set_xlabel("Root depth (m)")
ax.set_ylabel("Residuals")
plt.show()

 
 
#------Topography: elevation 
# What is the relationship between the elevation  and runoff ratio across catchments in the UK in 2015 
#Hypothesis: Catchments at higher elevations will have increased runoff

# creating a correlation plot 

elev_mean= final_table["elev_mean"]
fig, ax = plt.subplots(figsize=(5, 4))

ax.scatter(elev_mean,rr,s=25,alpha=0.3,color="tab:blue")
ax.set_xlabel("Mean elevation (m)")
ax.set_ylabel("Runoff ratio")
plt.show()


fig, ax = plt.subplots(figsize=(5, 4))

points = ax.scatter(elev_mean,rr,s=30,alpha=0.3,c =final_table["gauge_lat"] , cmap="managua")

plt.colorbar(points, ax=ax, label = "Latitude")
ax.set_xlabel("Mean elevation (m)")
ax.set_ylabel("Runoff ratio")
plt.show()


 # pearsons correlation rank 
stats.pearsonr(elev_mean, rr, method= None) 

 # fitting a linear regression 
slope, intercept = np.polyfit(elev_mean,rr, 1)

rr_predicted = intercept + slope * elev_mean

fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(elev_mean, rr, s=25, alpha=0.4, label="Catchments")
ax.plot(elev_mean, rr_predicted, color="black", lw=2, label="Linear regression")
ax.set_xlabel("Mean elevation (m)")
ax.set_ylabel("Runoff ratio")
ax.legend()
plt.show()

 #Assessing quality 

ss_res = np.sum((rr-rr_predicted)**2)
ss_tot = np.sum((rr - rr.mean())**2)
r2 = 1 - ss_res/ss_tot
print(r2)

residuals = (rr - rr_predicted)

print(residuals.describe())

fig, ax = plt.subplots(figsize=(5, 4))

ax.scatter(elev_mean, residuals, s=25, alpha=0.4)
ax.axhline(0, color="black", lw=1)
ax.set_xlabel("Mean elevation (m)")
ax.set_ylabel("Residuals")
plt.show()

 
