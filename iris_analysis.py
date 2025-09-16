
"""
Iris dataset analysis script.

This script:
- loads iris.csv
- creates several visualizations using matplotlib (and includes seaborn examples commented)
- saves PNG files for each plot.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

DATA_PATH = "iris.csv"
df = pd.read_csv(DATA_PATH)

# Quick cleanup / column names (ensure consistency)
df.columns = [c.strip() for c in df.columns]

# 1) Scatter plot: petal_length vs petal_width colored by species (matplotlib)
fig, ax = plt.subplots(figsize=(6,5))
species = df['species'].unique()
markers = ['o','s','^']
for sp, m in zip(species, markers):
    sub = df[df['species']==sp]
    ax.scatter(sub['petal_length'], sub['petal_width'], label=sp, marker=m)
ax.set_xlabel('Petal Length')
ax.set_ylabel('Petal Width')
ax.set_title('Petal Length vs Petal Width by Species')
ax.legend()
plt.tight_layout()
plt.savefig("plot_petal_scatter.png")
plt.close()

# 2) Boxplots for each numeric feature by species
num_cols = ['sepal_length','sepal_width','petal_length','petal_width']
for col in num_cols:
    fig, ax = plt.subplots(figsize=(6,4))
    data_to_plot = [df[df['species']==sp][col] for sp in species]
    ax.boxplot(data_to_plot, labels=species)
    ax.set_title(f'Boxplot of {col} by species')
    ax.set_ylabel(col)
    plt.tight_layout()
    fname = f"boxplot_{col}.png"
    plt.savefig(fname)
    plt.close()

# 3) Violin-like plot using matplotlib (density estimation per species for petal_length)
from scipy.stats import gaussian_kde
fig, ax = plt.subplots(figsize=(6,5))
x_min, x_max = df['petal_length'].min()-0.5, df['petal_length'].max()+0.5
x_grid = np.linspace(x_min, x_max, 200)
width_scale = 0.4
positions = np.arange(len(species)) + 1
for pos, sp in zip(positions, species):
    vals = df[df['species']==sp]['petal_length'].values
    kde = gaussian_kde(vals)
    density = kde(x_grid)
    # Normalize for plotting
    density = density / density.max() * width_scale
    ax.fill_betweenx(x_grid, pos - density, pos + density, alpha=0.6)
    ax.plot(np.full_like(x_grid, pos), x_grid, alpha=0)  # dummy to ensure axis limits
ax.set_yticks(np.linspace(x_min+0.5, x_max-0.5, 5))
ax.set_xticks(positions)
ax.set_xticklabels(species)
ax.set_xlabel('Species')
ax.set_ylabel('Petal Length')
ax.set_title('Violin-like density of Petal Length by Species')
plt.tight_layout()
plt.savefig("violin_petal_length.png")
plt.close()

# 4) Pairwise scatter matrix (simple)
from pandas.plotting import scatter_matrix
fig = plt.figure(figsize=(8,8))
axes = scatter_matrix(df[num_cols], alpha=0.7, diagonal='kde', figsize=(8,8))
# color points by species manually
colors = {'setosa':'red','versicolor':'green','virginica':'blue'}
for i, ax in enumerate(axes.flatten()):
    if ax.get_xlabel() in num_cols and ax.get_ylabel() in num_cols:
        for sp, col in colors.items():
            mask = df['species']==sp
            ax.scatter(df.loc[mask, ax.get_xlabel()], df.loc[mask, ax.get_ylabel()], label=sp, s=10)
plt.suptitle('Scatter matrix of numeric features (colored by species)')
plt.tight_layout()
plt.savefig("pairwise_scatter_matrix.png")
plt.close()

# Save a CSV summary
summary = df.groupby('species')[num_cols].agg(['mean','std','min','max'])
summary.to_csv("summary_by_species.csv")
print("Plots and summary saved.")
