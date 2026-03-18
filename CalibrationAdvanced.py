# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 22:27:54 2026

@author: nyala
"""

# PHOTOMETRIC CALIBRATION USING SIMBAD CATALOGUE STARS

# This script reads the spreadsheet produced by the photometry
# code, uses stars with catalogue magnitudes from SIMBAD to
# determine the calibration constants, and includes the
# instrumental uncertainties when fitting the constants.

import pandas as pd               # Used to read the Excel spreadsheet
import numpy as np                # Used for numerical calculations
import matplotlib.pyplot as plt   # Used to produce plots


# LOAD THE SPREADSHEET


# Read the Excel file containing the photometry output
df = pd.read_excel("reducedimagesM34.xlsx", sheet_name=1)

# Keep only the columns needed for calibration
df = df[["V","sigma_V","Mag V","B","sigma_B","Mag B"]]

# Remove rows where catalogue magnitudes are missing
# These are the stars that were not matched to SIMBAD
df = df.dropna(subset=["Mag B","Mag V"])


# EXTRACT DATA FROM THE TABLE
# Instrumental magnitudes measured from the CCD images
v = df["V"].to_numpy(dtype=float)
b = df["B"].to_numpy(dtype=float)

# Uncertainties in the instrumental magnitudes
sigma_v = df["sigma_V"].to_numpy(dtype=float)
sigma_b = df["sigma_B"].to_numpy(dtype=float)

# Catalogue magnitudes from SIMBAD
V_cat = df["Mag V"].to_numpy(dtype=float)
B_cat = df["Mag B"].to_numpy(dtype=float)


# CALCULATE INSTRUMENTAL COLOUR
# Instrumental colour index
color_inst = b - v

# Uncertainty in colour
sigma_color = np.sqrt(sigma_b**2 + sigma_v**2)


# FIT B-BAND CALIBRATION CONSTANTS
# Calibration equation:

# B_cat = b + B0 − alpha(b − v)

# Rearranged:

# y = B_cat − b = B0 − alpha*(b−v)

y_B = B_cat - b
x = color_inst

# Use instrumental B uncertainty as weight
weights_B = 1 / sigma_b**2

# Construct the design matrix
X_B = np.column_stack([-x, np.ones(len(x))])

# Compute weighted least squares solution
XTWX = X_B.T @ np.diag(weights_B) @ X_B
XTWy = X_B.T @ np.diag(weights_B) @ y_B

params_B = np.linalg.solve(XTWX, XTWy)

alpha = params_B[0]
B0 = params_B[1]

# Covariance matrix of the fitted parameters
cov_B = np.linalg.inv(XTWX)

# Uncertainties of the parameters
sigma_alpha = np.sqrt(cov_B[0,0])
sigma_B0 = np.sqrt(cov_B[1,1])


# FIT V-BAND CALIBRATION CONSTANTS
# Calibration equation:
# V_cat = v + V0 + beta(b − v)

y_V = V_cat - v

# Use instrumental V uncertainty as weight
weights_V = 1 / sigma_v**2

# Design matrix
X_V = np.column_stack([x, np.ones(len(x))])

XTWX = X_V.T @ np.diag(weights_V) @ X_V
XTWy = X_V.T @ np.diag(weights_V) @ y_V

params_V = np.linalg.solve(XTWX, XTWy)

beta = params_V[0]
V0 = params_V[1]

# Covariance matrix and uncertainties
cov_V = np.linalg.inv(XTWX)

sigma_beta = np.sqrt(cov_V[0,0])
sigma_V0 = np.sqrt(cov_V[1,1])


# PRINT CALIBRATION CONSTANTS WITH UNCERTAINTIES

print("alpha =", alpha, "+/-", sigma_alpha)
print("beta  =", beta, "+/-", sigma_beta)
print("B0    =", B0, "+/-", sigma_B0)
print("V0    =", V0, "+/-", sigma_V0)


# APPLY CALIBRATION TO ALL STARS

# Convert instrumental magnitudes to calibrated magnitudes
B_cal = b + B0 - alpha*(b - v)
V_cal = v + V0 + beta*(b - v)

BV_cal = B_cal - V_cal


# PLOT CALIBRATED COLOUR–MAGNITUDE DIAGRAM

plt.figure(figsize=(6,8))

plt.scatter(BV_cal, V_cal, s=10)

plt.xlabel("B - V")
plt.ylabel("V")
plt.title("Calibrated Colour-Magnitude Diagram")

plt.gca().invert_yaxis()

plt.grid(alpha=0.3)

plt.show()


# PLOT INSTRUMENTAL VS CATALOGUE MAGNITUDES

plt.figure(figsize=(6,6))
# Create a new square figure

plt.scatter(V_cat, v)
# Plot instrumental V (y-axis) against catalogue V (x-axis)

# Generate smooth x values for the fitted line
V_line = np.linspace(min(V_cat), max(V_cat), 100)

# Instrumental V predicted from the calibration equation
v_fit = V_line - V0 - beta*(b.mean() - v.mean())
# This approximates the calibration relation for plotting

plt.plot(V_line, v_fit)
# Draw the fitted line through the points

plt.xlabel("Catalogue V")
# Label x-axis

plt.ylabel("Instrumental V")
# Label y-axis

plt.title("Instrumental vs Catalogue V")

plt.grid(alpha=0.3)

plt.show()

#For B

plt.figure(figsize=(6,6))

plt.scatter(B_cat, b)
# Plot instrumental B vs catalogue B

B_line = np.linspace(min(B_cat), max(B_cat), 100)

b_fit = B_line - B0 + alpha*(b.mean() - v.mean())
# Predicted instrumental magnitude from calibration equation

plt.plot(B_line, b_fit)

plt.xlabel("Catalogue B")
plt.ylabel("Instrumental B")

plt.title("Instrumental vs Catalogue B")

plt.grid(alpha=0.3)

plt.show()




# PLOT M34 CMD USING THREE DIFFERENT CALIBRATION CONSTANT SETS
# This code applies three different calibration solutions
# (derived from M35, NGC752 and M34 standard stars)
# to the same instrumental M34 photometry and compares
# the resulting CMDs on a single plot.

import pandas as pd
# Pandas is used to read the Excel spreadsheet.

import numpy as np
# NumPy is used for numerical calculations.

import matplotlib.pyplot as plt
# Matplotlib is used to make the CMD plot.


# LOAD ALL STARS FROM THE SPREADSHEET
# This reads the full spreadsheet, not just the calibration stars.
# We only need the instrumental B and V columns here because the
# calibration constants have already been calculated separately.

df_all = pd.read_excel("reducedimagesM34.xlsx", sheet_name=1)

# Keep only the instrumental magnitude columns needed for the CMD
df_all = df_all[["B", "V"]].dropna()

# Extract instrumental B and V magnitudes for ALL stars
b = df_all["B"].to_numpy(dtype=float)
v = df_all["V"].to_numpy(dtype=float)

# Calculate instrumental colour for all stars
color_inst = b - v


# CALIBRATION CONSTANTS

# Constants derived using M35 calibration stars
alpha_M35 = -0.14160781979774323
beta_M35  = -0.02633050936171181
B0_M35    = 21.541161602688568
V0_M35    = 22.006693872983977

# Constants derived using NGC752 calibration stars
alpha_N752 = -0.22441440534176674
beta_N752  = 0.023418141843125
B0_N752    = 21.466527881876424
V0_N752    = 21.987255785163295

# Constants derived using M34 calibration stars
alpha_M34 = 0.20250657389566923
beta_M34  = -0.1989724887224234
B0_M34    = 21.601933954836223
V0_M34    = 21.991515788702973


# APPLY EACH CALIBRATION TO ALL STARS

# M35 calibration applied to all stars 
B_cal_M35 = b + B0_M35 - alpha_M35 * color_inst
V_cal_M35 = v + V0_M35 + beta_M35 * color_inst
BV_M35 = B_cal_M35 - V_cal_M35

# NGC752 calibration applied to all stars 
B_cal_N752 = b + B0_N752 - alpha_N752 * color_inst
V_cal_N752 = v + V0_N752 + beta_N752 * color_inst
BV_N752 = B_cal_N752 - V_cal_N752

# M34 calibration applied to all stars 
B_cal_M34 = b + B0_M34 - alpha_M34 * color_inst
V_cal_M34 = v + V0_M34 + beta_M34 * color_inst
BV_M34 = B_cal_M34 - V_cal_M34


# PLOT THE THREE CALIBRATED CMDs ON ONE GRAPH

plt.figure(figsize=(7, 8))

# Plot each calibrated CMD in a different colour
plt.scatter(BV_M35, V_cal_M35, s=10, color='blue', label="Calibration from M35")
plt.scatter(BV_N752, V_cal_N752, s=10, color='green', label="Calibration from NGC 752")
plt.scatter(BV_M34, V_cal_M34, s=10, color='red', label="Calibration from M34")



# Label the axes
plt.xlabel("B - V")
plt.ylabel("V")

# Invert the y-axis because brighter magnitudes are smaller numbers
plt.gca().invert_yaxis()

# Add a legend to show which colour corresponds to which calibration
plt.legend()

# Add a light grid to make the plot easier to read
plt.grid(alpha=0.3)

# Add a title
#plt.title("M34 Colour–Magnitude Diagram using Different Calibration Solutions")

# Display the plot
plt.show()


#covariance

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


# Best-fit parameters from your calibration
alpha = -0.14160781979774323
B0 = 21.541161602688568

# Uncertainties from your fit
sigma_alpha = 0.0005338422463729507
sigma_B0 = 0.00040376491834063464

# Covariance between parameters
# If you didn't store it, you can approximate it as 0
cov_alpha_B0 = 0

# Construct covariance matrix
cov = np.array([
    [sigma_alpha**2, cov_alpha_B0],
    [cov_alpha_B0, sigma_B0**2]
])


# Compute eigenvalues and eigenvectors
vals, vecs = np.linalg.eigh(cov)

# Sort eigenvalues
order = vals.argsort()[::-1]
vals = vals[order]
vecs = vecs[:, order]

# Angle of ellipse
theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

# Width and height of ellipse (1σ region)
width, height = 2 * np.sqrt(vals)


# Plot
fig, ax = plt.subplots()

ellipse = Ellipse(
    (alpha, B0),
    width,
    height,
    angle=theta,
    edgecolor='red',
    facecolor='none',
    lw=2
)

ax.add_patch(ellipse)

# Plot best-fit point
ax.scatter(alpha, B0, color='black')

plt.xlabel("alpha")
plt.ylabel("B0")
plt.title("Calibration Parameter Confidence Ellipse")

plt.show()