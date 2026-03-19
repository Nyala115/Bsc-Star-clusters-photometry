# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 22:27:54 2026

@author: nyala
"""

# PHOTOMETRIC CALIBRATION USING SIMBAD CATALOGUE STARS



import pandas as pd                     # For reading Excel data
import numpy as np                      # For numerical calculations
import matplotlib.pyplot as plt         # For plotting
from scipy.optimize import curve_fit    # For least-squares fitting



# LOAD THE DATA


# Read the Excel file containing photometry results
df = pd.read_excel("reducedimagesNGC752.xlsx", sheet_name=1)

# Keep only the columns needed for calibration
df = df[["V","sigma_V","Mag V","B","sigma_B","Mag B"]]

# Remove rows where catalogue magnitudes are missing
# (these are unmatched stars and cannot be used for calibration)
df = df.dropna(subset=["Mag B","Mag V"])



# EXTRACT DATA ARRAYS

# Instrumental magnitudes measured from the CCD
v = df["V"].to_numpy(dtype=float)
b = df["B"].to_numpy(dtype=float)

# Uncertainties in instrumental magnitudes
sigma_v = df["sigma_V"].to_numpy(dtype=float)
sigma_b = df["sigma_B"].to_numpy(dtype=float)

# Catalogue magnitudes (true values from SIMBAD)
V_cat = df["Mag V"].to_numpy(dtype=float)
B_cat = df["Mag B"].to_numpy(dtype=float)



# COMPUTE INSTRUMENTAL COLOUR

# Instrumental colour index (used in calibration equations)
color_inst = b - v

# Propagate uncertainties: σ² = σ_b² + σ_v²
sigma_color = np.sqrt(sigma_b**2 + sigma_v**2)


# DEFINE FITTING MODELS

# B-band calibration model:
# B_cat = b + B0 − alpha*(b − v)
# Rearranged to:
# y = B_cat − b = B0 − alpha*x

def model_B(x, alpha, B0):
    return B0 - alpha * x


# V-band calibration model:
# V_cat = v + V0 + beta*(b − v)
# Rearranged to:
# y = V_cat − v = V0 + beta*x

def model_V(x, beta, V0):
    return V0 + beta * x



# FIT B-BAND CALIBRATION


# Define variables for fitting
x = color_inst                      # Independent variable: (b − v)
y_B = B_cat - b                     # Dependent variable

# Perform weighted least-squares fit using curve_fit
popt_B, pcov_B = curve_fit(
    model_B,                        # Model function
    x,                              # x data
    y_B,                            # y data
    sigma=sigma_b,                  # Uncertainties in y
    absolute_sigma=True             # Use true uncertainties (important!)
)

# Extract fitted parameters
alpha, B0 = popt_B

# Extract parameter uncertainties from covariance matrix
sigma_alpha, sigma_B0 = np.sqrt(np.diag(pcov_B))



# FIT V-BAND CALIBRATION

y_V = V_cat - v

popt_V, pcov_V = curve_fit(
    model_V,
    x,
    y_V,
    sigma=sigma_v,
    absolute_sigma=True
)

beta, V0 = popt_V
sigma_beta, sigma_V0 = np.sqrt(np.diag(pcov_V))



# PRINT RESULTS

print("Calibration constants:")
print("alpha =", alpha, "+/-", sigma_alpha)
print("beta  =", beta, "+/-", sigma_beta)
print("B0    =", B0, "+/-", sigma_B0)
print("V0    =", V0, "+/-", sigma_V0)



# APPLY CALIBRATION

# Convert instrumental magnitudes into calibrated magnitudes
B_cal = b + B0 - alpha * (b - v)
V_cal = v + V0 + beta * (b - v)

# Compute calibrated colour index
BV_cal = B_cal - V_cal



# PLOT COLOUR–MAGNITUDE DIAGRAM

plt.figure(figsize=(6,8))

# Scatter plot of CMD
plt.scatter(BV_cal, V_cal, s=10)

# Axis labels
plt.xlabel("B - V")
plt.ylabel("V")

# Invert y-axis (bright stars have smaller magnitudes)
plt.gca().invert_yaxis()

plt.title("Calibrated Colour-Magnitude Diagram")

plt.grid(alpha=0.3)

plt.show()


# PLOT CALIBRATION RELATIONS

#  V band plot 
plt.figure(figsize=(6,6))

plt.scatter(V_cat, v)

# Generate smooth line
V_line = np.linspace(min(V_cat), max(V_cat), 100)

# Plot approximate fit line
v_fit = V_line - V0 - beta * np.mean(color_inst)

plt.plot(V_line, v_fit)

plt.xlabel("Catalogue V")
plt.ylabel("Instrumental V")

plt.title("Instrumental vs Catalogue V")

plt.grid(alpha=0.3)

plt.show()


#  B band plot
plt.figure(figsize=(6,6))

plt.scatter(B_cat, b)

B_line = np.linspace(min(B_cat), max(B_cat), 100)

b_fit = B_line - B0 + alpha * np.mean(color_inst)

plt.plot(B_line, b_fit)

plt.xlabel("Catalogue B")
plt.ylabel("Instrumental B")

plt.title("Instrumental vs Catalogue B")

plt.grid(alpha=0.3)

plt.show()