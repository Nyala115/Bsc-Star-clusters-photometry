# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 06:54:16 2026

@author: nyala
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

#performed on 3 different files, M35, M34 and NGC752
df = pd.read_excel("M35output.xlsx", sheet_name=1)
#reads excel file

print(df.columns)
#check columns

df_V = df[["V", "sigma_V", "Mag V"]].dropna()
#selects specified columns ,.dropna ignores missing value rows

v = df_V["V"].to_numpy()
sigma_v = df_V["sigma_V"].to_numpy()
V = df_V["Mag V"].to_numpy()
#converts each column into numpy arrays

df_B = df[["B", "sigma_B", "Mag B"]].dropna()
#same as v

b = df_B["B"].to_numpy()
sigma_b = df_B["sigma_B"].to_numpy()
B = df_B["Mag B"].to_numpy()
#same as v

def model(x, X0):
    return x - X0
#defines function , v= V - V0 as y= x - X0
#x is catalogue magnitude
#X0 is zero point offset
#y is instrumental magnitude


# V band fit
popt_V, pcov_V = curve_fit(model, V, v, sigma=sigma_v)
V0 = popt_V[0]
#popt_v is best-fit parameters
#pcov_v is covariance matrix (uncertainty of fit)
#model is function to fit (defined above)

# B band fit
popt_B, pcov_B = curve_fit(model, B, b, sigma=sigma_b)
B0 = popt_B[0]

print("V0 =", V0)
print("B0 =", B0)
#displays calibration constants

import matplotlib.pyplot as plt

plt.errorbar(V, v, yerr=sigma_v, fmt='o') #yerr is vertical error bars
xfit = np.linspace(min(V), max(V), 100) #100 evenly spaced points between the smallest and largest catalog magntiudes
plt.plot(xfit, xfit - V0) #plots fitted model
plt.xlabel("Catalogue V")
plt.ylabel("Instrumental v")
plt.show() 


V_cal = df["V"] + V0
B_cal = df["B"] + B0
#applies calibration

BV_cal = B_cal - V_cal
#calibrated B -V

plt.figure(figsize=(6,8))

plt.scatter(BV_cal, V_cal, s=10)

plt.xlabel("B - V")
plt.ylabel("V")
plt.title("Calibrated CMD")

plt.gca().invert_yaxis()   # Important for magnitudes
plt.grid(alpha=0.3) #adds cool grid lines

plt.show()
