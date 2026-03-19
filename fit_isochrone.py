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

sigma_psf = 8.4 
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


#print("Image alignment delta_x, delta_y = ",
#      "{:7.2f}".format(delta_x), "{:7.2f}".format(delta_y), "\n")
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


 
#isochrone fitting

import read_mist_models

# LOAD THE MIST ISOCHRONE GRID
# This reads the theoretical isochrone file from disk.
# The file contains many stellar evolution tracks for different ages.

isocmd = read_mist_models.ISOCMD(
    'MIST_v1.2_feh_m0.25_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd '
)
#MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd
#MIST_v1.2_feh_p0.25_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd
#MIST_v1.2_feh_m0.25_afe_p0.0_vvcrit0.0_UBVRIplus.iso.cmd 
# CHOOSE FIT PARAMETERS MANUALLY
# These are the values to adjust by hand until the plotted
# isochrone gives the best visual match to the observed CMD.

age = 8.55
mu  = 11.4
Av  = 0.95 # because E(B-V) ~ 0.20 and Rv = 3.1
Rv  = 3.1      # standard Galactic extinction law

# Reddening in colour:
# E(B-V) = A_V / R_V
EBminusV = Av / Rv


# OBSERVED CLUSTER DATA
# BminusV_cal = calibrated colour
# V_cal       = calibrated V magnitude

colour = np.array(BminusV_cal, dtype=float)
mag = np.array(V_cal, dtype=float)


# CLEAN THE OBSERVED CMD
# These cuts remove obvious outliers and keep only the region
# of the CMD relevant to the cluster sequence.

obs_mask = (
    np.isfinite(colour) &
    np.isfinite(mag) &
    (colour > 0.0) & (colour < 1.5) &
    (mag > 9.0) & (mag < 16.5)
)

colour = colour[obs_mask]
mag = mag[obs_mask]


# OPTIONAL: DEFINE A FITTING SUBSET
# This narrower region is used only for the residual and
# turn-off scatter calculation.
# It helps avoid outliers and obvious field-star contamination.

fit_mask = (
    (colour > 0.1) & (colour < 0.85) &
    (mag > 10.0) & (mag < 14.5)
)

colour_fit = colour[fit_mask]
mag_fit = mag[fit_mask]

print("Number of stars after obs_mask =", len(colour))
print("Number of stars after fit_mask =", len(colour_fit))


# EXTRACT THE CHOSEN ISOCHRONE
# age is given as log10(age / years), so this finds the nearest
# age in the MIST grid.

age_ind = isocmd.age_index(age)
iso = isocmd.isocmds[age_ind]

# Extract theoretical B and V magnitudes from the isochrone
B_iso = np.array(iso['Bessell_B'], dtype=float)
V_iso = np.array(iso['Bessell_V'], dtype=float)

# Compute theoretical colour
BV_iso = B_iso - V_iso


# REMOVE INVALID MODEL POINTS
# This ensures only finite values are used in the plot and
# later interpolation.

good = np.isfinite(BV_iso) & np.isfinite(V_iso)
BV_iso = BV_iso[good]
V_iso = V_iso[good]


# APPLY REDDENING AND DISTANCE SHIFTS
# Horizontal shift:
#   (B-V)_obs = (B-V)_iso + E(B-V)

# Vertical shift:
#   V_obs = V_iso + mu + A_V

BV_shift = BV_iso + EBminusV
V_shift = V_iso + mu + Av


# PLOT OBSERVED CMD + MANUAL ISOCHRONE
# This is the main visual fitting plot.
# Change age, mu and Av above, rerun, and inspect the match.

plt.figure(figsize=(6, 8))

# Plot observed stars
plt.scatter(BminusV_cal, V_cal, s=10, color='black', label='Observed')

# Plot chosen isochrone
plt.plot(BV_shift, V_shift, color='red',
         label=f"log age = {age:.2f}, mu = {mu:.2f}")

# Standard CMD convention: brighter stars at the top
plt.gca().invert_yaxis()

plt.xlabel("B - V")
plt.ylabel("V")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# PREPARE ISOCHRONE FOR INTERPOLATION
# This part is only needed to estimate the scatter of the fit.

# Keep only the CMD region relevant to the observed sequence
use = (
    (BV_shift > 0.0) & (BV_shift < 1.5) &
    (V_shift > 8.0) & (V_shift < 17.0)
)

BV_fit = BV_shift[use]
V_fit = V_shift[use]

# Sort by colour so interpolation works properly
order = np.argsort(BV_fit)
BV_fit = BV_fit[order]
V_fit = V_fit[order]

# Remove duplicate colour values
BV_fit_unique, idx = np.unique(np.round(BV_fit, 4), return_index=True)
V_fit_unique = V_fit[idx]


# INTERPOLATE THE ISOCHRONE AT THE OBSERVED STAR COLOURS
# This gives the model V magnitude at each observed colour.

iso_interp = np.interp(
    colour_fit,
    BV_fit_unique,
    V_fit_unique,
    left=np.nan,
    right=np.nan
)

valid = np.isfinite(iso_interp)

residuals = mag_fit[valid] - iso_interp[valid]
colour_valid = colour_fit[valid]


# COMPUTE TURN-OFF SCATTER
# This estimates how well the chosen isochrone matches the
# main-sequence turn-off region.

turnoff = (colour_valid > 0.25) & (colour_valid < 0.75)

sigma_turnoff = np.nanstd(residuals[turnoff])

print("Chosen log(age) =", age)
print("Chosen age =", 10**age / 1e9, "Gyr")
print("Chosen distance modulus =", mu)
print("Distance =", 10**((mu + 5)/5), "pc")
print("Turnoff scatter =", sigma_turnoff)
print("Number of stars used in fit =", len(colour_fit))
print("Number of valid interpolated stars =", np.sum(valid))

