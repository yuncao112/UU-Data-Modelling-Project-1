# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:12:16 2026

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

x = np.log10(final_table["conductivity_hypres"])
y = final_table["runoff_ratio"]

slope, intercept = np.polyfit(x, y, 1)

fig, ax = plt.subplots(figsize=(5, 4))

ax.scatter(x,y,s = 25, alpha = 0.3)
ax.plot(x, intercept + slope * x)

ax.set_xlabel("log10 Hydraulic conductivity")
ax.set_ylabel("Runoff ratio")
plt.show()

r, p = stats.pearsonr(x, y)
r2 = r**2

print(f"Correlation coefficient : {r:.3f}")
print(f"p-value                 : {p:.2e}")
print(f"R²                      : {r2:.3f}")

import statsmodels.formula.api as smf

model_arid = smf.ols(
    "runoff_ratio ~ aridity",
    data=final_table).fit()

intercept = model_arid.params["Intercept"]
slope = model_arid.params["aridity"]

print(model_arid.summary())

x = final_table["aridity"]
y = final_table["runoff_ratio"]

fig, ax = plt.subplots(figsize=(5, 4))

ax.scatter(x, y, s=25, alpha=0.3)
ax.plot(x, intercept + slope * x)

ax.set_xlabel("Aridity")
ax.set_ylabel("Runoff ratio")

plt.show()

final_table["log_conductivity"] = np.log10(final_table["conductivity_hypres"])

model_arid_logcon = smf.ols(
    "runoff_ratio ~ aridity + log_conductivity",
    data=final_table).fit()

print(model_arid_logcon.summary())

y_obs = model_arid_logcon.model.endog
y_pred = model_arid_logcon.predict()

plt.figure(figsize=(6,6))
plt.scatter(y_obs, y_pred, s=25,alpha=0.6,color="tab:blue")
plt.plot([y_obs.min(), y_obs.max()],[y_obs.min(), y_obs.max()],'k--')
plt.xlabel("Observed runoff ratio")
plt.ylabel("Predicted runoff ratio")
plt.show()

fitted = model_arid_logcon.fittedvalues
residuals = model_arid_logcon.resid
plt.scatter(fitted, residuals, s=25, alpha=0.6)
plt.axhline(0, linestyle="--")

plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.show()

plt.hist(residuals, bins=30)

plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

limit = max(abs(residuals.min()), abs(residuals.max()))

plt.figure(figsize = [5,7])
plt.scatter(
    final_table["gauge_lon"],
    final_table["gauge_lat"],
    c=residuals,
    cmap="coolwarm",
    vmin=-limit,
    vmax=limit,
)

plt.colorbar(label="Residual")

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# add log-transformed conductivity to final_table
final_table["log_conductivity"] = np.log10(final_table["conductivity_hypres"])


#climate + soil + land cover
model_3var = smf.ols(
    "runoff_ratio ~ aridity + log_conductivity + urban_perc_2015",
    data=final_table).fit()

print(model_3var.summary())


#try another soil variable
model_3var_tawc = smf.ols(
    "runoff_ratio ~ aridity + tawc + urban_perc_2015",
    data=final_table).fit()

print(model_3var_tawc.summary())

y_obs = model_3var.model.endog
y_pred = model_3var.predict()

plt.figure(figsize=(6,6))
plt.scatter(y_obs, y_pred, s=25, alpha=0.6, color="tab:blue")
plt.plot(
    [y_obs.min(), y_obs.max()],
    [y_obs.min(), y_obs.max()],
    'k--'
)

plt.xlabel("Observed runoff ratio")
plt.ylabel("Predicted runoff ratio")
plt.show()

# Residual analysis

fitted = model_3var.fittedvalues
residuals = model_3var.resid

plt.scatter(fitted, residuals, s=25, alpha=0.6)
plt.axhline(0, linestyle="--")

plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.show()


plt.hist(residuals, bins=30)

plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

# Mapping residuals

limit = max(abs(residuals.min()), abs(residuals.max()))

plt.figure(figsize=[5,7])
plt.scatter(
    final_table["gauge_lon"],
    final_table["gauge_lat"],
    c=residuals,
    cmap="coolwarm",
    vmin=-limit,
    vmax=limit,
)

plt.colorbar(label="Residual")

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

climate_vars = [
    "t_mean",
    "pet_mean",
    "p_mean",
    "elev_mean",
]

climate_data = final_table[climate_vars].dropna()

climate_data.head()

# Pearson correlation matrix
corr = final_table[climate_vars].corr()
print("Correlation matrix: \n", corr.round(2))

# Plot
plt.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
plt.colorbar(label="Correlation")

plt.xticks(range(len(climate_vars)), climate_vars, rotation=45, ha="right")
plt.yticks(range(len(climate_vars)), climate_vars)

plt.title("Correlation matrix")
plt.show()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
climate_scaled = scaler.fit_transform(climate_data)

from sklearn.decomposition import PCA

pca = PCA()

climate_pca = pca.fit_transform(climate_scaled)

explained_variance = pca.explained_variance_ratio_

plt.figure(figsize=(6,4))

plt.plot(
    range(1, len(explained_variance)+1),
    explained_variance,
    marker="o"
)

plt.xticks(range(1, len(explained_variance)+1))
plt.xlabel("Principal component")
plt.ylabel("Fraction of explained variance")

plt.show()

print(explained_variance)

plt.figure(figsize=(7,5))

plt.scatter(
    climate_pca[:,0],
    climate_pca[:,1]
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid()
plt.show()

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[
        "PC1",
        "PC2",
        "PC3",
        "PC4",
    ],
    index=climate_vars
)
loadings

plt.scatter(
    climate_pca[:,0],
    climate_pca[:,1],
    alpha=0.6
)

for i, var in enumerate(climate_vars):
    plt.arrow(
        0, 0,
        loadings.iloc[i,0],
        loadings.iloc[i,1],
        color="black",
        head_width=0.05,
        length_includes_head=False
    )

    plt.text(
        loadings.iloc[i,0]*1.1,
        loadings.iloc[i,1]*1.1,
        var,
        color="black"
    )

plt.xlabel(f"PC1 ({explained_variance[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({explained_variance[1]*100:.1f}%)")

plt.axhline(0, color="grey", linewidth=0.5)
plt.axvline(0, color="grey", linewidth=0.5)

plt.show()

lat = final_table.loc[climate_data.index, "gauge_lat"]

plt.figure(figsize=(7,5))

sc = plt.scatter(
    climate_pca[:,0],
    climate_pca[:,1],
    c=lat,
    cmap="viridis"
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(sc, label="Latitude")
plt.grid()

plt.show()

lon = final_table.loc[climate_data.index, "gauge_lon"]

plt.figure(figsize=(7,5))

sc = plt.scatter(
    climate_pca[:,0],
    climate_pca[:,1],
    c=lon,
    cmap="viridis"
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(sc, label="Longitude")
plt.grid()

plt.show()

pca_vars = [
    "t_mean",
    "pet_mean",
    "p_mean",
    "soil_depth_pelletier",
    "conductivity_hypres",
    "root_depth",
    "porosity_hypres",
    "urban_perc_2015",
    "dwood_perc_2015",
    "ewood_perc_2015",
    "grass_perc_2015",
    "crop_perc_2015",
    "elev_mean",
    "dpsbar"
]

catchment_data = final_table[pca_vars].dropna()

catchment_data.head()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
catchment_scaled = scaler.fit_transform(catchment_data)

from sklearn.decomposition import PCA

pca_all = PCA()

catchment_pca = pca_all.fit_transform(catchment_scaled)

explained_variance_all = pca_all.explained_variance_ratio_

plt.figure(figsize=(7,5))

plt.plot(
    range(1, len(explained_variance_all)+1),
    explained_variance_all,
    marker="o"
)

plt.xticks(range(1, len(explained_variance_all)+1))
plt.xlabel("Principal component")
plt.ylabel("Fraction of explained variance")
plt.show()

print(explained_variance_all)

cumulative_variance = np.cumsum(explained_variance_all)

plt.figure(figsize=(7,5))

plt.plot(
    range(1, len(cumulative_variance)+1),
    cumulative_variance,
    marker="o"
)

plt.axhline(0.8, linestyle="--")
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative explained variance")
plt.xticks(range(1, len(cumulative_variance)+1))

plt.show()

print(cumulative_variance)

loadings_all = pd.DataFrame(
    pca_all.components_.T,
    columns=[f"PC{i+1}" for i in range(len(pca_vars))],
    index=pca_vars
)

loadings_all[["PC1", "PC2"]]

plt.figure(figsize=(7,5))

plt.scatter(
    catchment_pca[:,0],
    catchment_pca[:,1],
    alpha=0.6
)

plt.xlabel(f"PC1 ({explained_variance_all[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({explained_variance_all[1]*100:.1f}%)")
plt.grid()

plt.show()

runoff_pca = final_table.loc[catchment_data.index, "runoff_ratio"]
print("PC1 vs runoff ratio:",
      np.corrcoef(catchment_pca[:,0], runoff_pca)[0,1])

print("PC2 vs runoff ratio:",
      np.corrcoef(catchment_pca[:,1], runoff_pca)[0,1])

plt.figure(figsize=(7,5))

sc = plt.scatter(
    catchment_pca[:,0],
    catchment_pca[:,1],
    c=runoff_pca,
    cmap="viridis",
    alpha=0.7
)

plt.xlabel(f"PC1 ({explained_variance_all[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({explained_variance_all[1]*100:.1f}%)")

plt.colorbar(sc, label="Runoff ratio")

plt.show()

# Loadings of PC1 and PC2
loadings_table = pd.DataFrame({
    "PC1": pca_all.components_[0],
    "PC2": pca_all.components_[1]
}, index=pca_vars)

print(loadings_table.round(3))