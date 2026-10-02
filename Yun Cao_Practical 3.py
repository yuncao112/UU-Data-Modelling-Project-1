# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 11:09:55 2026

@author: 12619
"""

# import packages
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# directory settings
PROJECT_DIR   = Path("D:/DaMod-computer-labs/week-02-data-wrangling")
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
FIGURE_DIR    = PROJECT_DIR / "figures"

final_table_path = PROCESSED_DIR / "camels_gb_1991_2020_analysis_ready.csv"

# Load the processed CAMELS-GB table created in the previous practical
final_table = pd.read_csv(final_table_path)

# Inspect the first rows
final_table.head()

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.hist(final_table["runoff_ratio"], bins=30, color="tab:blue", edgecolor="white")
ax.set_xlabel("Runoff ratio")
ax.set_ylabel("Number of catchments")
ax.set_title("Distribution of runoff ratio")
plt.show()

rr = final_table["runoff_ratio"]

mean_rr = rr.mean()
median_rr = rr.median()
sigma_rr = rr.std(ddof=1)

print(f"Mean              : {mean_rr:.3f}")
print(f"Median            : {median_rr:.3f}")
print(f"Standard deviation: {sigma_rr:.3f}")

fig, ax = plt.subplots(figsize=(6, 3.5))

# Histogram as probability density
ax.hist(rr, bins=30, color="tab:blue", edgecolor="white",
    alpha=0.9, density=True, label="Observed distribution")

# Create x-values spanning the data range
x_rr = np.linspace(rr.min(), rr.max(), 200)
# Calculate the normal probability density function
pdf_rr = stats.norm.pdf(x_rr, mean_rr, sigma_rr)

# Plot the fitted normal distribution
ax.plot(x_rr, pdf_rr, color="black", linewidth=2, label="Normal distribution")

ax.set_xlabel("Runoff ratio")
ax.set_ylabel("Probability density")
ax.legend()
plt.show()

# Get q_mean data
q = final_table["q_mean"]

# RAW q_mean

mean_q = q.mean()
median_q = q.median()
sigma_q = q.std(ddof=1)

print(f"Mean              : {mean_q:.3f}")
print(f"Median            : {median_q:.3f}")
print(f"Standard deviation: {sigma_q:.3f}")


fig, ax = plt.subplots(figsize=(6, 3.5))

# Histogram as probability density
ax.hist(q, bins=30, color="tab:blue", edgecolor="white",
        alpha=0.9, density=True, label="Observed distribution")

# x-values
x_q = np.linspace(q.min(), q.max(), 200)

# Normal distribution with same mean and standard deviation
pdf_q = stats.norm.pdf(x_q, mean_q, sigma_q)

# Plot fitted normal distribution
ax.plot(x_q, pdf_q, color="black", linewidth=2,
        label="Normal distribution")

ax.set_xlabel("Mean discharge")
ax.set_ylabel("Probability density")
ax.set_title("Distribution of q_mean")
ax.legend()

plt.show()


# LOG-TRANSFORMED q_mean

# q_mean must be positive before log10 transformation
q_positive = q[q > 0]

log_q = np.log10(q_positive)

mean_log_q = log_q.mean()
median_log_q = log_q.median()
sigma_log_q = log_q.std(ddof=1)

print(f"Mean              : {mean_log_q:.3f}")
print(f"Median            : {median_log_q:.3f}")
print(f"Standard deviation: {sigma_log_q:.3f}")


fig, ax = plt.subplots(figsize=(6, 3.5))

# Histogram
ax.hist(log_q, bins=30, color="tab:blue", edgecolor="white",
        alpha=0.9, density=True, label="Observed distribution")

# x-values
x_log_q = np.linspace(log_q.min(), log_q.max(), 200)

# Normal PDF
pdf_log_q = stats.norm.pdf(x_log_q, mean_log_q, sigma_log_q)

# Plot fitted normal distribution
ax.plot(x_log_q, pdf_log_q, color="black", linewidth=2,
        label="Normal distribution")

ax.set_xlabel("log10(Mean discharge)")
ax.set_ylabel("Probability density")
ax.set_title("Distribution of log10(q_mean)")
ax.legend()

plt.show()

# Sample size
n = len(rr)

# Standard error of the mean
se_rr = sigma_rr/ np.sqrt(n)

print(f"Number of catchments: {n}")
print(f"Standard error: {se_rr:.4f}")

ci_rr = stats.t.interval(
    confidence=0.95,
    df= n - 1, # Degrees of freedom
    loc=mean_rr,
    scale=se_rr
)

print(f"95% confidence interval: {ci_rr[0]:.3f} - {ci_rr[1]:.3f}")

fig, ax = plt.subplots(figsize=(6.5, 3.5))
ax.hist(rr, bins=30, color="tab:blue", edgecolor="white",
    density=True, alpha=0.9, label="Observed distribution")

ax.plot(x_rr, pdf_rr, color="black", linewidth=2, label="Normal distribution")

# Mean and Confidence interval
ax.axvline(mean_rr, color="tab:red", lw=2,label="Sample mean")
ax.axvspan(ci_rr[0], ci_rr[1], color="tab:red", alpha=0.3, label="95% confidence interval")

ax.set_xlabel("Runoff ratio")
ax.set_ylabel("Probability density")
ax.set_title("Runoff ratio distribution and uncertainty in the mean")
ax.legend()
plt.show()

# Mean discharge data
q = final_table["q_mean"]

# Number of catchments
n_q = len(q)

# Mean and standard deviation
mean_q = q.mean()
sigma_q = q.std(ddof=1)

# Standard error
se_q = sigma_q / np.sqrt(n_q)

# 95% confidence interval
ci_lower_q = mean_q - 1.96 * se_q
ci_upper_q = mean_q + 1.96 * se_q

print(f"Number of catchments: {n_q}")
print(f"Mean discharge: {mean_q:.3f}")
print(f"Standard deviation: {sigma_q:.3f}")
print(f"Standard error: {se_q:.4f}")
print(f"95% confidence interval: {ci_lower_q:.3f} - {ci_upper_q:.3f}")

# Relative uncertainty
relative_uncertainty_q = se_q / mean_q
relative_uncertainty_rr = 0.0089 / 0.538

print(f"Relative uncertainty of q_mean: {relative_uncertainty_q:.3%}")
print(f"Relative uncertainty of runoff ratio: {relative_uncertainty_rr:.3%}")

arid = final_table["aridity"]
fig, ax = plt.subplots(figsize=(5, 4))

ax.scatter(arid,rr,s=25,alpha=0.3,color="tab:blue")
ax.set_xlabel("Aridity index (PET / P)")
ax.set_ylabel("Runoff ratio")
plt.show()

r, p = stats.pearsonr(arid,rr)
print(f"Pearson correlation coefficient: {r:.3f}")
print(f"p-value: {p:.3e}")

fig, ax = plt.subplots(figsize=(5, 4))

points = ax.scatter(arid,rr, c=final_table["gauge_lat"], s=30, alpha=0.5, cmap="managua")

plt.colorbar(points, ax=ax, label="Latitude")
ax.set_xlabel("Aridity index (PET / P)")
ax.set_ylabel("Runoff ratio")
plt.show()

# Landcover
data_urban = final_table[
    ["urban_perc_2015", "runoff_ratio", "gauge_lat"]
].dropna()

urban = data_urban["urban_perc_2015"]
rr_urban = data_urban["runoff_ratio"]

# Scatter plot
fig, ax = plt.subplots(figsize=(5, 4))

ax.scatter(urban, rr_urban, s=25, alpha=0.3, color="tab:blue")
ax.set_xlabel("Urban percentage (%)")
ax.set_ylabel("Runoff ratio")

plt.show()


# Pearson correlation
r, p = stats.pearsonr(urban, rr_urban)

print(f"Urban percentage")
print(f"Pearson correlation coefficient: {r:.3f}")
print(f"p-value: {p:.3e}")


# Colour points by latitude
fig, ax = plt.subplots(figsize=(5, 4))

points = ax.scatter(
    urban,
    rr_urban,
    c=data_urban["gauge_lat"],
    s=30,
    alpha=0.5,
    cmap="managua"
)

plt.colorbar(points, ax=ax, label="Latitude")
ax.set_xlabel("Urban percentage (%)")
ax.set_ylabel("Runoff ratio")

plt.show()

# Soil attribute
data_tawc = final_table[
    ["tawc", "runoff_ratio", "gauge_lat"]
].dropna()

tawc = data_tawc["tawc"]
rr_tawc = data_tawc["runoff_ratio"]

# Scatter plot
fig, ax = plt.subplots(figsize=(5, 4))

ax.scatter(tawc, rr_tawc, s=25, alpha=0.3, color="tab:blue")
ax.set_xlabel("TAWC")
ax.set_ylabel("Runoff ratio")

plt.show()


# Pearson correlation
r, p = stats.pearsonr(tawc, rr_tawc)

print(f"TAWC")
print(f"Pearson correlation coefficient: {r:.3f}")
print(f"p-value: {p:.3e}")


# Colour points by latitude
fig, ax = plt.subplots(figsize=(5, 4))

points = ax.scatter(
    tawc,
    rr_tawc,
    c=data_tawc["gauge_lat"],
    s=30,
    alpha=0.5,
    cmap="managua"
)

plt.colorbar(points, ax=ax, label="Latitude")
ax.set_xlabel("TAWC")
ax.set_ylabel("Runoff ratio")

plt.show()

# Topographical attribute
data_dpsbar = final_table[
    ["dpsbar", "runoff_ratio", "gauge_lat"]
].dropna()

dpsbar = data_dpsbar["dpsbar"]
rr_dpsbar = data_dpsbar["runoff_ratio"]

# Scatter plot
fig, ax = plt.subplots(figsize=(5, 4))

ax.scatter(dpsbar, rr_dpsbar, s=25, alpha=0.3, color="tab:blue")
ax.set_xlabel("DPSBAR")
ax.set_ylabel("Runoff ratio")

plt.show()


# Pearson correlation
r, p = stats.pearsonr(dpsbar, rr_dpsbar)

print(f"DPSBAR")
print(f"Pearson correlation coefficient: {r:.3f}")
print(f"p-value: {p:.3e}")


# Colour points by latitude
fig, ax = plt.subplots(figsize=(5, 4))

points = ax.scatter(
    dpsbar,
    rr_dpsbar,
    c=data_dpsbar["gauge_lat"],
    s=30,
    alpha=0.5,
    cmap="managua"
)

plt.colorbar(points, ax=ax, label="Latitude")
ax.set_xlabel("DPSBAR")
ax.set_ylabel("Runoff ratio")

plt.show()

slope, intercept = np.polyfit(arid, rr, 1)

print(f"Slope = {slope:.3f}")
print(f"Intercept = {intercept:.3f}")

rr_predicted = intercept + slope * arid

fig, ax = plt.subplots(figsize=(5, 4))

ax.scatter(arid, rr, s=25, alpha=0.4, label="Catchments")
ax.plot(arid, rr_predicted, color="black", lw=2, label="Linear regression")

ax.set_xlabel("Aridity index (PET / P)")
ax.set_ylabel("Runoff ratio")
ax.legend()

plt.show()

ss_res = np.sum((rr - rr_predicted)**2)
ss_tot = np.sum((rr - rr.mean())**2)
r2 = 1 - ss_res / ss_tot

print(f"R² = {r2:.3f}")

# Difference between observed and predicted values
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

# Standard error of the slope
se_slope = np.sqrt(np.sum(residuals**2) /(n - 2)/np.sum((arid - arid.mean())**2))

# 95% confidence interval
t_value = stats.t.ppf(0.975, n - 2)

slope_lower = slope - t_value * se_slope
slope_upper = slope + t_value * se_slope

print(f"Slope estimate = {slope:.3f}")
print(f"95% confidence interval: {slope_lower:.3f} to {slope_upper:.3f}")

# Prepare data
data_dpsbar = final_table[
    ["dpsbar", "runoff_ratio"]
]

dpsbar = data_dpsbar["dpsbar"]
rr_dpsbar = data_dpsbar["runoff_ratio"]


# Scatter plot
fig, ax = plt.subplots(figsize=(5, 4))

ax.scatter(dpsbar, rr_dpsbar, s=25, alpha=0.4)

ax.set_xlabel("DPSBAR")
ax.set_ylabel("Runoff ratio")

plt.show()


# Linear regression
slope_dpsbar, intercept_dpsbar = np.polyfit(dpsbar, rr_dpsbar, 1)

print(f"Slope = {slope_dpsbar:.3f}")
print(f"Intercept = {intercept_dpsbar:.3f}")


# Predicted runoff ratio
rr_dpsbar_predicted = intercept_dpsbar + slope_dpsbar * dpsbar


# Plot regression line
fig, ax = plt.subplots(figsize=(5, 4))

ax.scatter(dpsbar, rr_dpsbar, s=25, alpha=0.4, label="Catchments")
ax.plot(dpsbar, rr_dpsbar_predicted,
        color="black", lw=2, label="Linear regression")

ax.set_xlabel("DPSBAR")
ax.set_ylabel("Runoff ratio")
ax.legend()

plt.show()


# R squared
ss_res = np.sum((rr_dpsbar - rr_dpsbar_predicted)**2)
ss_tot = np.sum((rr_dpsbar - rr_dpsbar.mean())**2)

r2_dpsbar = 1 - ss_res / ss_tot

print(f"R² = {r2_dpsbar:.3f}")

# Residuals
residuals_dpsbar = rr_dpsbar - rr_dpsbar_predicted


fig, ax = plt.subplots(figsize=(5, 4))

ax.scatter(dpsbar, residuals_dpsbar, s=25, alpha=0.4)
ax.axhline(0, color="black", lw=1)

ax.set_xlabel("DPSBAR")
ax.set_ylabel("Residuals")

plt.show()


# Histogram of residuals
fig, ax = plt.subplots(figsize=(6, 3.5))

ax.hist(residuals_dpsbar, bins=30, edgecolor="white")
ax.axvline(0, color="black", lw=1)

ax.set_xlabel("Residuals")
ax.set_ylabel("Number of catchments")
ax.set_title("Distribution of regression residuals")

plt.show()