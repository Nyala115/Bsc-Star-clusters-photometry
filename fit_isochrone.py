# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 15:56:21 2026

@author: nyala
"""

import read_mist_models
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from sfLib import *
import sys
import numpy as np
import scipy.stats as stats
from astropy.io import fits
from os.path import exists
from collections import deque
import matplotlib.patches as patches
from datetime import date
plt.rcParams["font.size"] = 12
#first section of this script up until isochrone fitting is sf.main provided and written by professor G. Cowan
imageFileDir_V = r'c:\users\nyala\onedrive\documents\rhul\bsc project\Data_project\\'
imageFileBase_V = 'M37_2026-03-02_19-37-09_V_0040'
imageFileDir_B = r'c:\users\nyala\onedrive\documents\rhul\bsc project\Data_project\\'
imageFileBase_B = 'M37_2026-03-02_19-36-35_B_0039'

#m34_2026-01-04_17-11-20_B_0019
#m34_2026-01-04_17-11-55_V_0020
#M37_2026-03-02_19-36-35_B_0039
#M37_2026-03-02_19-37-09_V_0040
#M35_2026-03-02_19-27-34_B_0029
#M35_2026-03-02_19-28-09_V_0030

# Parameters for image analysis (by default same for both images)
box = 401     
 #size of square region for backround est. Large box enough pixels to statistically determine background.
 #must be large  compared to PSF (has to be large comapared to spread of stars light or its dominated) 
 #but small comapred to image scale so background doesnt vary in box 
 
hole = 41    
#cuts  a  central hole for star so its excluded from background 
                           
border = np.array([2500, 2500, 1000, 1000])  
#defines unusable image edges. tells program "dont trust edges"
#because CCD suffers from vignetting, thermal gradients.
#vignetting is when light from stars doesnt reach every part of detector equally well.
#Light comes in at steeper angles, some is blocked or clipped by filters, lenses ect.
#so fewer photons reach detector at edges

cornerRadius = 1000   
#rounds corners of the usable region. removes irregular edge artefacts. 
#further removes effects of vignetting as it effects corners and edges most
#can be seen in the colour bar plot  
                   
threshold_factor = 9 #changed    
#pixels must exceed background by 5 stdv to count as signal.  
  
useExistingBackground = True 
#reuse background files if they exist

backgroundAlg = 'gauss'  
#selects Gaussian fitting method
#gaussian fits are more robust than medians  
#this is because they focus more on the peak and largely ignore stars and outliers
#Gaussian models shape of background peak. controlled mainly by where most pixels are
#median shifts if there are many stars and background isnt symmetric, also ignores info about width of noise distr
  
minPix = 8 #changed!!!!!! 
# must be across 6 pixels as 1 or 2 is likely to be noise/hot pixels
                   
sigma_cl_min = 6   #changed!!!!       
sigma_cl_max = 10     #changed!!!!
#sigma_cl is a measure of width of detected cluster of pixels.
#sigma min says if its less than 6 dont consider it a star as it could be noise 
#sigma max says if its more than 10 dont consider it as it could be 2 stars or background galaxies, nebulae etc
#PSF is around 8 pixels so within this range.  
      
rho_cl_max = 0.15 #changed!!!!!  
#rho_clo measures how elongated a detector object is (how round or stretched a cluster of pixels are)  
#low rho is good (round)
#high rho is bad (stretched or line)
#so only accept objects close to circular
    
clusterCut = np.array([minPix, sigma_cl_min, sigma_cl_max, rho_cl_max])
#puts all boundaries/limits together

sigma_psf = 8.0 
#how wide typical star looks in image
#sigma used as stars dont look like dots they look like bell shaped blobs
#describe "blob" with gaussian with sigma showing width
              
apR = 24 
#when measuring star brightness add up light inside circle of radius 16
#most star light is within 2 sigma so radius of 16 is 2x8 so good
#for gaussian distr 95% is within 2sigma.
#anything higher could may find more signal but significantly more noise.
#this is why we used 99th percentile in mini project.
                  
makeSpreadsheet = True 
#save result to spreadsheet file.



sf_V = StarField(imageFileDir_V, imageFileBase_V)
#Open the image and prepare it for star finding and photometry.
# starfield is a class defined in sfLib.py(a blue print for an object)
#sf_V is the object
#() info used to create it



sf_V.setUsableRegion(border, cornerRadius)
sf_V.estimateBackground(box, hole, backgroundAlg, useExistingBackground,
                        imageFileDir_V)
# Extract the background image estimated by Starfinder

# NYALA Histogram of raw pixel values from entire V image
pixels = sf_V.n.flatten()

plt.figure()
plt.hist(pixels, bins=300, histtype='step', color='red')
plt.yscale('log')
plt.xlabel('Pixel counts (ADU)')
plt.ylabel('Number of pixels')
plt.title('Histogram of raw pixel counts (entire V image)')
plt.show()

#NYALA START
# Histogram of pixel counts from one 401x401 background box

center_row = 1500     # change this to move the box vertically
center_col = 1000     # change this to move the box horizontally
halfBox = 200        # still gives 401x401 box


# Extract the box
boxPixels = sf_V.n[
    center_row - halfBox : center_row + halfBox + 1, #this before the comma selects rows
    center_col - halfBox : center_col + halfBox + 1 #the second part after the comma selects columns
]


pixels = boxPixels.flatten()


# Plot histogram
#plt.figure()
#plt.hist(pixels, bins=300, histtype='step', color='red')
#plt.yscale('log')
#plt.xlabel('Pixel counts (ADU)')
#plt.ylabel('Number of pixels')
#plt.title(f'Pixel count histogram ({2*halfBox+1}×{2*halfBox+1} box at ({center_row},{center_col}))')
#plt.show()

#END OF NYALA


sf_V.findClusters(threshold_factor)
#finds groups of pixels significantly brighter than background that could be stars
#A cluster is a group of neighbouring bright pixels above the threshold etc.
#each cluster is a "candidate star"
#find clusters uses estimate background to decide what counts as "real"

sf_V.processClusterList(clusterCut)      
#From all the detected clusters, keep only the ones that look like real stars.
#uses cluster cut to identify from boundaries set what is a a star
# applies this to the detected clusters to make sure detections are trusted


# Repeat for B image
sf_B = StarField(imageFileDir_B, imageFileBase_B)
sf_B.setUsableRegion(border, cornerRadius)
sf_B.estimateBackground(box, hole, backgroundAlg, useExistingBackground,
                        imageFileDir_B)
sf_B.findClusters(threshold_factor)
sf_B.processClusterList(clusterCut)
#just same process but for B


#  Align images
default = 0     # brightest star in image, index 0 corresponds to brightest
prompt = ("Star no. for alignment from first image [" + str(default) + "]: ")
#creates a string called prompt,asks user what star number to use

istar_V = int(input(prompt) or default)
#displays prompt, waits for user input,converts input into integer, if 'enter' is pressed uses default
#must use same star in both images to allign them

prompt = ("Star no. for alignment from second image [" + str(default) + "]: ")
#same but for second image

istar_B = int(input(prompt) or default)
#same but for second image

delta_x = sf_B.muxList[istar_B] - sf_V.muxList[istar_V]
#calculates horizontal shift between images
#muxlist stores the x-coordinate(column position) of each detected star in pixel units
#sf_B.muxlist[istar_B] x position in the B image and same for V
#Subtraction tells us how far the B imahe is shifted relative to the V image in x

delta_y = sf_B.muyList[istar_B] - sf_V.muyList[istar_V]
#same but for y
#both x and y required for accurate alignment 


print("Image alignment delta_x, delta_y = ",
      "{:7.2f}".format(delta_x), "{:7.2f}".format(delta_y), "\n")
#prints calculated offsets in a neat, readable format
#{:7.2f} width of 7 characters, 2 digits decimal point

#Alignment is important before you compare V and B brightness, must insure its measuring the same star in each image
#Finds offset in images, allows code to shift and enables accurate colour (B-V) measurement
#without this stars would be mismatched, colours would be wrong, CMD would be meaningless



# Photometry on star locations from V image, adjusted V positions for B image
muList = zip(sf_V.muxList, sf_V.muyList)
#creates a list of (x, y) coordinate pairs for all detcted stars in V image
#muxlist is x positions and muylist is y positions
# (x1, y1), (x2, y2).... each pair corresponds to one stars position

V, sigma_V = apPhot(sf_V, muList, apR, sigma_psf)
#meaures V-band brightness of every star and estimates the uncertainty
#apPhot is aperture photometry. Aperture- a cirular region drawn around star. Photometry- measuring light
#for each position in mulist, apPhot : 
    #draws circular aperture of radius apR
    #sums pixel values inside the aperture
    #subtracts local backg
    #applies PSF-based corrections
    #estimates measurement uncertainty
# V -- array of V magnitudes, sigma_V -- corresponding uncertainties

muList = zip(sf_V.muxList + delta_x, sf_V.muyList + delta_y)
#creates a new list of star positions, shifted to match B-band image
#zip takes two or more lists and ties them together element by element, eg:
    #x=[1, 2, 3] y=[1, 2, 3]
    #list(zip(x, y)) gives (1,1), (2,2), (3,3)
    
    
B, sigma_B = apPhot(sf_B, muList, apR, sigma_psf)
#measures B-band brightness of the same stars
#same process as before


#calculates color index for each star. measures stellar colour, correlates to temp

sigma_BminusV = np.sqrt(sigma_V**2 + sigma_B**2)
#uncertainty in B-V using standard error propagation

# Apply calibration
# Full colour-term calibration 
alpha = -0.08647598193297154
beta  = 0.02485162956586614
B0    = 21.59642415746947
V0    = 21.957844147377465

color_inst = B - V   # instrumental (b - v)

B_cal = B + B0 - alpha * color_inst
V_cal = V + V0 + beta  * color_inst
BminusV_cal = B_cal - V_cal


 

# Isochrone fitting

import read_mist_models


# Load the MIST isochrone grid from file.

# MIST provides theoretical stellar evolution models. For a given age and
# chemical composition, it predicts the colours and magnitudes of stars
# along an isochrone, which is the locus of stars of the same age on a
# colour–magnitude diagram (CMD).
isocmd = read_mist_models.ISOCMD(
    'MIST_v1.2_feh_p0.25_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd'
)

# Observed stellar colour and magnitude arrays.

# BminusV_cal contains calibrated colour values:
#   (B - V) = B magnitude - V magnitude

# V_cal contains calibrated apparent V magnitudes.

# dtype=float ensures the arrays are numerical and suitable for later
# interpolation and fitting calculations.
colour = np.array(BminusV_cal, dtype=float)
mag = np.array(V_cal, dtype=float)


# Reddening parameters

# Visual extinction in magnitudes.

# Extinction is caused by interstellar dust, which absorbs and scatters
# starlight. This makes stars appear dimmer.

# A_V is the extinction in the V band.
Av = 0.5

# Standard ratio of total to selective extinction.


#   R_V = A_V / E(B-V)

# where E(B-V) is the colour excess, also called reddening.
Rv = 3.1

# Colour excess derived from the extinction law:
#   E(B-V) = A_V / R_V

# Reddening shifts stars horizontally to the right on the CMD because dust
# absorbs blue light more strongly than red light.
EBminusV = Av / Rv


# Clean observed CMD

# Create a mask selecting only stars with:
# - finite colour and magnitude values
# - colour inside a reasonable CMD range
# - magnitude inside a useful observed range

# np.isfinite removes NaN and infinite values.

# The colour and magnitude limits exclude extreme outliers, poor
# measurements, and objects outside the cluster sequence region of
# interest.
obs_mask = (
    np.isfinite(colour) &
    np.isfinite(mag) &
    (colour > 0.0) & (colour < 1.5) &
    (mag > 9.0) & (mag < 16.5)
)

# Apply the mask so that only stars inside the selected CMD region remain.
colour = colour[obs_mask]
mag = mag[obs_mask]


# Define the subset of stars used for fitting

# This mask is slightly narrower than the full observed CMD region.

# The purpose is to fit only the part of the CMD where the cluster sequence
# is clearest and least contaminated by outliers, field stars, or poorly
# measured stars.
fit_mask = (
    (colour > 0.1) & (colour < 1.4) &
    (mag > 9.5) & (mag < 16.5)
)

# Arrays containing only the stars actually used in the fitting process.
colour_fit = colour[fit_mask]
mag_fit = mag[fit_mask]

# Print the number of stars remaining after each cut.
print("Number of stars after obs_mask =", len(colour))
print("Number of stars after fit_mask =", len(colour_fit))


# Helper function to prepare one isochrone for fitting

def get_isochrone(age, mu):
    """
    Return a processed isochrone for a given age and distance modulus.

    INPUTS:
      age : log10(age / years)
      mu  : distance modulus

    OUTPUTS:
      BV_unique : reddened and shifted isochrone colour values
      V_unique  : reddened and shifted isochrone V magnitudes

    PHYSICS:
    An isochrone gives absolute magnitudes and intrinsic colours predicted
    by stellar evolution theory for stars of a single age.

    To compare with observations, two shifts are needed:
      1. Reddening/extinction from interstellar dust
      2. Distance modulus to convert absolute magnitude to apparent magnitude

    The relation for apparent magnitude is:
        m = M + mu + A

    where:
      m  = apparent magnitude
      M  = absolute magnitude
      mu = distance modulus
      A  = extinction in that band
    """

    # Find the index in the MIST grid corresponding to the requested age.
    age_ind = isocmd.age_index(age)

    # Extract the isochrone table for that age.
    iso = isocmd.isocmds[age_ind]

    # Extract model B and V magnitudes.
    B_iso = np.array(iso['Bessell_B'], dtype=float)
    V_iso = np.array(iso['Bessell_V'], dtype=float)

    # Compute model colour:
    #   (B - V) = B magnitude - V magnitude
    BV_iso = B_iso - V_iso

    # Keep only finite values.
    good = np.isfinite(BV_iso) & np.isfinite(V_iso)
    BV_iso = BV_iso[good]
    V_iso = V_iso[good]

    # Apply reddening in colour:
    #   (B - V)_observed = (B - V)_intrinsic + E(B-V)
    BV_iso = BV_iso + EBminusV

    # Apply distance modulus and extinction in magnitude:
    #   V_observed = V_intrinsic + mu + A_V
    V_iso = V_iso + Av + mu

    # Keep only the CMD region relevant to the observed cluster sequence.
    
    # This avoids fitting parts of the theoretical isochrone far outside
    # the observed data range.
    use = (
        (BV_iso > 0.0) & (BV_iso < 1.5) &
        (V_iso > 8.5) & (V_iso < 17.0)
    )

    BV_iso = BV_iso[use]
    V_iso = V_iso[use]

    # Sort by colour so that interpolation works correctly.
    
    # np.interp requires the x-array to be monotonic.
    order = np.argsort(BV_iso)
    BV_iso = BV_iso[order]
    V_iso = V_iso[order]

    # Remove duplicate colour values.
    
    # Interpolation behaves better when each x-value appears only once.
    # Rounding to 4 decimal places avoids tiny floating-point differences.
    BV_unique, idx = np.unique(np.round(BV_iso, 4), return_index=True)
    V_unique = V_iso[idx]

    return BV_unique, V_unique


# Fit grid

# Select a grid of trial ages from the available MIST ages.

# MIST ages are stored as log10(age / years).
# For example:
#   log(age) = 8.0  means 10^8 years
#   log(age) = 9.0  means 10^9 years
age_grid = [age for age in isocmd.ages if 8.0 <= age <= 9.2]

# Trial distance moduli.

# The distance modulus is defined by:
#   mu = m - M = 5 log10(d / 10 pc)

# so it determines the vertical shift of the isochrone on the CMD.
mu_grid = np.linspace(9.0, 12.5, 150)

# Initialise best-fit values.
best_chi2 = np.inf
best_age = None
best_mu = None

# Loop over all trial ages.
for age in age_grid:

    # Build the isochrone once to check whether enough valid points exist.
    BV_iso, V_iso = get_isochrone(age, 0.0)

    # Skip ages for which the usable isochrone section is too short.
    if len(BV_iso) < 20:
        continue

    # Loop over all trial distance moduli.
    for mu in mu_grid:

        # Build the reddened and shifted isochrone for this age and mu.
        BV_iso, V_shift = get_isochrone(age, mu)

        # Interpolate the isochrone magnitude at the observed star colours.
        
        # For each observed colour, np.interp gives the corresponding model
        # V magnitude on the isochrone.
        
        # left=np.nan and right=np.nan prevent extrapolation beyond the
        # colour range of the model.
        iso_interp = np.interp(
            colour_fit,
            BV_iso,
            V_shift,
            left=np.nan,
            right=np.nan
        )

        # Keep only stars where interpolation was successful.
        valid = np.isfinite(iso_interp)

        # If too few stars are valid, skip this fit.
        if np.sum(valid) < 5:
            continue

        # Residuals between observed and model magnitudes:
        #   residual = observed V - model V
        residuals = mag_fit[valid] - iso_interp[valid]

        # Mean squared residual used as the fitting statistic.
        
        # called chi^2 in the code.
        
        # Smaller values indicate a better fit.
        chi2 = np.mean(residuals**2)

        # Update the best fit if this trial is better.
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_age = age
            best_mu = mu

# If no valid fit was found, stop with an error message.
if best_age is None or best_mu is None:
    raise RuntimeError(
        "No valid isochrone fit found. Try widening the age/mu grid or "
        "loosening the CMD/isochrone masks."
    )

# Print the best-fit results.

# best_age is log10(age / years), so:
#   age in years = 10^best_age
#   age in Gyr   = 10^best_age / 1e9
print("Best log(age) =", best_age)
print("Best age =", 10**best_age / 1e9, "Gyr")

# Best-fit distance modulus.
print("Best distance modulus =", best_mu)

# Convert distance modulus to distance in parsecs using:
#   mu = 5 log10(d / 10)
# which rearranges to:
#   d = 10^((mu + 5)/5)
print("Distance =", 10**((best_mu + 5)/5), "pc")

# Best-fit mean squared residual.
print("Best chi^2 =", best_chi2)


# Plot observed CMD and best-fit isochrone

plt.figure()

# In a CMD, brighter stars have smaller magnitudes, so the y-axis is
# inverted to match the standard astronomical convention.
plt.gca().invert_yaxis()

# Plot all observed calibrated stars.
plt.scatter(BminusV_cal, V_cal, s=10, color='black', label='Observed')

# Use the best-fit age and distance modulus.
age = best_age
mu = best_mu

# Redefine reddening values for clarity in the plotting section.
Av = 0.5
Rv = 3.1
EBminusV = Av / Rv

# Find the model isochrone corresponding to the best-fit age.
age_ind = isocmd.age_index(age)

# Extract model B and V magnitudes.
B_iso = isocmd.isocmds[age_ind]['Bessell_B']
V_iso = isocmd.isocmds[age_ind]['Bessell_V']

# Compute model colour.
BV_iso = B_iso - V_iso

# Plot the shifted isochrone.
#
# Horizontal shift:
#   + E(B-V)
#
# Vertical shift:
#   + mu + A_V
plt.plot(
    BV_iso + EBminusV,
    V_iso + mu + Av,
    color='red',
    label=f"log age = {age:.2f}"
)

plt.xlabel("B - V")
plt.ylabel("V")
plt.legend()
plt.show()


# Turn-off scatter

# Rebuild the best-fit isochrone using the final parameters.
B_iso = np.array(isocmd.isocmds[age_ind]['Bessell_B'], dtype=float)
V_iso = np.array(isocmd.isocmds[age_ind]['Bessell_V'], dtype=float)
BV_iso = B_iso - V_iso

# Keep only finite values.
good = np.isfinite(BV_iso) & np.isfinite(V_iso)
BV_iso = BV_iso[good]
V_iso = V_iso[good]

# Apply reddening and distance/extinction shifts.
BV_fit = BV_iso + EBminusV
V_fit = V_iso + mu + Av

# Keep only the CMD region relevant to the observed sequence.
use = (
    (BV_fit > 0.0) & (BV_fit < 1.3) &
    (V_fit > 8.0) & (V_fit < 16.5)
)
BV_fit = BV_fit[use]
V_fit = V_fit[use]

# Sort by colour for interpolation.
order = np.argsort(BV_fit)
BV_fit = BV_fit[order]
V_fit = V_fit[order]

# Remove duplicate colour values.
BV_fit_unique, idx = np.unique(np.round(BV_fit, 4), return_index=True)
V_fit_unique = V_fit[idx]

# Interpolate the best-fit isochrone at the colours of the stars used in fitting.
iso_interp = np.interp(
    colour_fit,
    BV_fit_unique,
    V_fit_unique,
    left=np.nan,
    right=np.nan
)

# Keep only successful interpolation points.
valid = np.isfinite(iso_interp)

# Residuals between observed and fitted isochrone magnitudes.
residuals = mag_fit[valid] - iso_interp[valid]

# Colours of stars with valid interpolated values.
colour_valid = colour_fit[valid]

# Approximate turn-off region in colour.

# The main-sequence turn-off is the point where stars begin leaving the main
# sequence. The scatter in this region provides a simple measure of how well
# the isochrone matches the upper main sequence and can also reflect
# photometric scatter, field contamination, binaries, or imperfect reddening.
turnoff = (colour_valid > 0.25) & (colour_valid < 0.75)

# Standard deviation of residuals in the turn-off region.

# np.nanstd ignores NaN values.
# A smaller value means the fitted isochrone passes more closely through
# the observed turn-off stars.
sigma_turnoff = np.nanstd(residuals[turnoff])

# Print turn-off scatter diagnostics.
print("Turnoff scatter =", sigma_turnoff)
print("Number of stars used in fit =", len(colour_fit))
print("Number of valid interpolated stars =", np.sum(valid))


