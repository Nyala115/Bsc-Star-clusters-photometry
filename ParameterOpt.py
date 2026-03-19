# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:33:04 2026

@author: nyala
"""
from sfLib import *
from scipy.optimize import curve_fit
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from astropy.io import fits
from os.path import exists
from collections import deque
import matplotlib.patches as patches
from datetime import date
plt.rcParams["font.size"] = 12
#m34_2026-01-04_17-11-20_B_0019
#m34_2026-01-04_17-11-55_V_0020
#M37_2026-03-02_19-36-35_B_0039
#M37_2026-03-02_19-37-09_V_0040
#M35_2026-03-02_19-27-34_B_0029
#M35_2026-03-02_19-28-09_V_0030
#files used^^

#this code is based upon/uses sf.main.py provided by G.Cowan

imageFileDir_V = r'c:\users\nyala\onedrive\documents\rhul\bsc project\Data_project\\'
imageFileBase_V = 'm34_2026-01-04_17-11-55_V_0020'
imageFileDir_B = r'c:\users\nyala\onedrive\documents\rhul\bsc project\Data_project\\'
imageFileBase_B = 'm34_2026-01-04_17-11-20_B_0019'

#nyala
def estimate_psf(sf):
   """
   Estimate the stellar PSF width from accepted stars using the
   covariance matrices already computed by Starfinder.

   """
    # List to store width estimates for stars that pass all quality cuts.
    sigma_list = []

    # If no stars were accepted, the PSF cannot be estimated from the data.
    # A fallback value of 8 pixels is returned.
    if len(sf.starList) == 0:
        print("WARNING: No accepted stars found")
        return 8.0

    # Brightness threshold set relative to the brightest accepted star.
    # Faint stars have noisier shape measurements, so only stars brighter
    # than one tenth of the brightest star are used.
    bright_flux_limit = sf.starList[0].flux() / 10.0

    # Minimum allowed distance to the nearest neighbouring star, in pixels.
    # This reduces the effect of overlapping stellar profiles.
    min_separation = 60.0

    # Loop over all accepted stars.
    for star in sf.starList:

        # Total flux of the star.
        flux = star.flux()

        # Covariance matrix describing the shape of the light distribution.
        cov = star.cov()

        # The diagonal terms represent variances, so they must be positive.
        # Non-physical values are skipped.
        if cov[0, 0] <= 0 or cov[1, 1] <= 0:
            continue

        # RMS width in the x-direction.
        sigma_x = np.sqrt(cov[0, 0])

        # RMS width in the y-direction.
        sigma_y = np.sqrt(cov[1, 1])

        # Correlation coefficient derived from the covariance matrix.
        # Values close to zero indicate a more symmetric profile.
        rho_xy = cov[0, 1] / (sigma_x * sigma_y)

        # Conditions for selecting a good PSF star:
        #1. sufficiently bright
        #2. sufficiently isolated
        #3. nearly circular / not strongly elongated
        good_star = (
            flux > bright_flux_limit and
            sf.nearestDist[star.clusterNum] > min_separation and
            abs(rho_xy) < 0.1
        )

        if good_star:
            # Mean of the x and y widths gives one representative width.
            sigma_mean = 0.5 * (sigma_x + sigma_y)

            # Extremely small or large widths are rejected as unrealistic.
            if 3.0 < sigma_mean < 15.0:
                sigma_list.append(sigma_mean)

    # If no stars survive the cuts, return the fallback PSF width.
    if len(sigma_list) == 0:
        print("WARNING: No valid PSF stars found")
        return 8.0

    # The median is used rather than the mean because it is less sensitive
    # to outliers.
    sigma_psf = np.median(sigma_list)

    # For a Gaussian profile:
    #   FWHM = 2*sqrt(2*ln2)*sigma ≈ 2.355*sigma
    # FWHM is a standard astronomical measure of image quality.
    fwhm = 2.355 * sigma_psf

    # Diagnostic output.
    print("\nNumber of stars used for PSF:", len(sigma_list))
    print("PSF sigma values:", sigma_list)
    print("Estimated PSF sigma =", sigma_psf)
    print("Estimated PSF FWHM =", fwhm)

    # Histogram of the sigma values used in the PSF estimate.
    plt.figure()
    plt.hist(sigma_list, bins=20, histtype='step', color='blue')
    plt.axvline(sigma_psf, color='red', linestyle='--', label='Median PSF sigma')

    plt.xlabel("PSF sigma (pixels)")
    plt.ylabel("Number of stars")
    plt.title("Distribution of stars used for PSF estimate")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    return sigma_psf


# Parameters for image analysis

# Size of the square region used for local background estimation.
# This box must be much larger than the PSF so that individual stars do
# not dominate, but small enough that background variations across the
# image are still local.
box = 401

# Size of the central hole excluded from the background box.
# This prevents a star near the centre of the box from biasing the
# background estimate upward.
hole = 41

# Width of excluded image borders: [left, right, bottom, top].
# The detector edges are often less reliable because of vignetting,
# flat-field uncertainties, reduced illumination, and other edge effects.
border = np.array([2500, 2500, 1000, 1000])

# Radius used to round off the corners of the usable region.
# Corners often show stronger vignetting and more artefacts than the
# central part of the image.
cornerRadius = 1000

# Detection threshold in units of the background standard deviation.
# A pixel is considered significant if:
#   pixel > background + threshold_factor * sigma_background
# A larger threshold reduces false detections from noise but can miss
# faint stars.
threshold_factor = 8

# Existing background files are reused if available.
# This saves time and keeps repeated runs consistent.
useExistingBackground = True

# Background estimation algorithm.
# 'gauss' means the pixel histogram is fitted with a Gaussian.
# This is useful because the sky background noise is often approximately
# Gaussian near its peak.
backgroundAlg = 'gauss'

# Minimum number of connected above-threshold pixels required for a
# detection to be accepted.
# Real stellar images cover several pixels because of the PSF, whereas
# hot pixels or noise spikes are often much smaller.
minPix = 8

# Minimum and maximum allowed widths for accepted detections.
# Objects smaller than this are likely noise, and objects larger than this
# may be blends, extended objects, or poorly measured sources.
sigma_cl_min = 8
sigma_cl_max = 9

# Maximum allowed correlation coefficient for the cluster shape.
# Values close to zero correspond to more symmetric objects, while larger
# values indicate elongation or distortion.
rho_cl_max = 0.1

# Array containing the selection cuts passed to Starfinder.
clusterCut = np.array([minPix, sigma_cl_min, sigma_cl_max, rho_cl_max])

# Adopted PSF width in pixels.
# For a Gaussian profile, sigma describes the width of the stellar image.
sigma_psf = 8.0

# Aperture radius used for photometry.
# For a Gaussian light profile:
#   about 68% of the light lies within 1 sigma
#   about 95% lies within 2 sigma
#   about 99.7% lies within 3 sigma
# With sigma_psf = 8 pixels, an aperture radius of 24 pixels corresponds
# to 3 sigma and should therefore contain almost all of the stellar flux.
apR = 24.0

# Output measurements are saved to a spreadsheet if this is True.
makeSpreadsheet = True


# Create StarField object for the V image

# StarField is a class defined in sfLib.py.
# sf_V is one instance containing the V-band image and methods for
# background estimation, star finding, and photometry.
sf_V = StarField(imageFileDir_V, imageFileBase_V)

# Define the trusted usable region of the image.
sf_V.setUsableRegion(border, cornerRadius)

# Estimate the background across the image.
# This is required so that the detection threshold can be defined relative
# to the local background and noise level.
sf_V.estimateBackground(box, hole, backgroundAlg, useExistingBackground,
                        imageFileDir_V)


# Histogram of raw pixel values from the entire V image

# Flatten the 2D image array into 1D so that a histogram can be plotted.
pixels = sf_V.n.flatten()

plt.figure()
plt.hist(pixels, bins=300, histtype='step', color='red')

# A logarithmic y-axis is useful because the number of pixels can vary
# over several orders of magnitude across the histogram.
plt.yscale('log')

plt.xlabel('Pixel counts (ADU)')
plt.ylabel('Number of pixels')
plt.title('Histogram of raw pixel counts (entire V image)')
plt.show()


# Histogram of pixel values from one 401x401 background box

# Central coordinates of the selected box.
center_row = 1500
center_col = 1000

# Half-width of the box. The total width is 2*halfBox + 1 = 401 pixels.
halfBox = 200

# Extract the sub-image.
# The first slice selects rows and the second slice selects columns.
boxPixels = sf_V.n[
    center_row - halfBox: center_row + halfBox + 1,
    center_col - halfBox: center_col + halfBox + 1
]

# Flatten the 2D sub-image into a 1D array for histogram plotting.
pixels = boxPixels.flatten()

plt.figure()
plt.hist(pixels, bins=300, histtype='step', color='red')
plt.yscale('log')
plt.xlabel('Pixel counts (ADU)')
plt.ylabel('Number of pixels')
plt.title(f'Pixel count histogram ({2*halfBox+1}×{2*halfBox+1} box at ({center_row},{center_col}))')
plt.show()


# Star detection

# Find connected groups of pixels above the chosen threshold.
# Because of the PSF, a star is expected to cover several neighbouring
# bright pixels rather than a single pixel.
sf_V.findClusters(threshold_factor)

# Keep only detections that satisfy the pixel number, width and shape cuts.
# This removes many false detections and non-stellar objects.
sf_V.processClusterList(clusterCut)


# Histogram of detected star widths

# Store a representative width for each accepted star.
sigma_list = []

for star in sf_V.starList:
    cov = star.cov()

    # Only stars with valid positive variances are used.
    if cov[0, 0] > 0 and cov[1, 1] > 0:
        sigma_x = np.sqrt(cov[0, 0])
        sigma_y = np.sqrt(cov[1, 1])

        # Mean width of the object.
        sigma_mean = 0.5 * (sigma_x + sigma_y)
        sigma_list.append(sigma_mean)

if len(sigma_list) > 0:
    plt.figure()
    plt.hist(sigma_list, bins=25, histtype='step', color='blue')

    # Vertical lines mark the adopted lower and upper width cuts.
    plt.axvline(sigma_cl_min, color='red', linestyle='--', label='sigma_cl_min')
    plt.axvline(sigma_cl_max, color='red', linestyle='--', label='sigma_cl_max')

    plt.xlabel("Cluster width (pixels)")
    plt.ylabel("Number of accepted stars")
    plt.title("Distribution of detected star widths")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


# Histogram of accepted star fluxes

# Store fluxes of all accepted stars.
flux_list = []

for star in sf_V.starList:
    flux_list.append(star.flux())

if len(flux_list) > 0:
    plt.figure()
    plt.hist(flux_list, bins=30, histtype='step', color='green')

    # Stellar fluxes usually span a wide range, so a logarithmic x-axis
    # makes the distribution easier to interpret.
    plt.xscale('log')

    plt.xlabel("Star flux")
    plt.ylabel("Number of accepted stars")
    plt.title("Distribution of accepted star fluxes")
    plt.grid(alpha=0.3)
    plt.show()


# Shape parameter versus width

# Store width and shape information for each accepted star.
sigma_vals = []
rho_vals = []

for star in sf_V.starList:
    cov = star.cov()

    if cov[0, 0] > 0 and cov[1, 1] > 0:
        sigma_x = np.sqrt(cov[0, 0])
        sigma_y = np.sqrt(cov[1, 1])

        # Correlation coefficient derived from the covariance matrix.
        rho_xy = cov[0, 1] / (sigma_x * sigma_y)

        # Mean width of the object.
        sigma_mean = 0.5 * (sigma_x + sigma_y)

        sigma_vals.append(sigma_mean)
        rho_vals.append(rho_xy)

if len(sigma_vals) > 0:
    plt.figure()
    plt.scatter(sigma_vals, rho_vals, s=10)

    # Selection limits for width and shape.
    plt.axvline(sigma_cl_min, color='orange', linestyle='--', label='sigma_cl_min')
    plt.axvline(sigma_cl_max, color='orange', linestyle='--', label='sigma_cl_max')
    plt.axhline(rho_cl_max, color='red', linestyle='--', label='rho_cl_max')

    plt.xlabel("Cluster width (pixels)")
    plt.ylabel("Shape parameter rho")
    plt.title("Cluster shape parameter vs width")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


# Histogram of number of pixels per accepted detection

# numPix() gives the number of connected pixels belonging to each detection.
# This helps show whether the chosen minimum pixel cut is reasonable.
numPix_list = []

for star in sf_V.starList:
    numPix_list.append(star.numPix())

if len(numPix_list) > 0:
    plt.figure()
    plt.hist(numPix_list, bins=20, histtype='step', color='purple')

    # Vertical line marking the adopted lower cut.
    plt.axvline(minPix, color='red', linestyle='--', label='minPix')

    plt.xlabel("Number of pixels in detection")
    plt.ylabel("Number of accepted stars")
    plt.title("Distribution of detection sizes")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


# Detection statistics

# clusterList contains all threshold detections.
# starList contains only those detections that passed the quality cuts.
print("Number of detected clusters in V image =", len(sf_V.clusterList))
print("Number of accepted stars in V image   =", len(sf_V.starList))


# Spatial distribution of accepted stars

# Extract centroid positions of accepted stars.
x_vals = np.array(sf_V.muxList)
y_vals = np.array(sf_V.muyList)

plt.figure()
plt.scatter(x_vals, y_vals, s=10)

# This plot is useful for checking edge effects, crowding, gradients in
# detectability, and suspicious gaps across the detector.
plt.xlabel("X pixel")
plt.ylabel("Y pixel")
plt.title("Spatial distribution of accepted stars")
plt.grid(alpha=0.3)
plt.show()

# Example star image cutout

# Select the first accepted star in the list.
# In this code, this is assumed to be the brightest detected star.
star = sf_V.starList[0]

# Get the centroid position of the star.
# mu() returns the measured centre of the light distribution in pixel
# coordinates, usually as non-integer values because the centroid can lie
# between pixel centres.
mux, muy = star.mu()

# Convert centroid coordinates to integers so they can be used as array indices.
x = int(mux)
y = int(muy)

# Half-size of the square image cutout.
# A value of 20 gives a square extending 20 pixels in each direction from
# the centre, so the full cutout is approximately 40 × 40 pixels.
size = 20

# Extract a small sub-image around the star from the full image array.
# sf_V.n is the CCD image array.
# The slice [y-size:y+size, x-size:x+size] selects rows first and then columns.
cutout = sf_V.n[y-size:y+size, x-size:x+size]

# Create a new figure for the cutout image.
plt.figure()

# Display the cutout as an image.
# cmap='inferno' sets the colour scale.
# origin='lower' places the coordinate origin at the bottom left, which is
# often more natural for astronomical image display.
plt.imshow(cutout, cmap='inferno', origin='lower')

# Add a colour bar showing the pixel value scale.
plt.colorbar()

# Axis labels in pixel coordinates.
plt.xlabel("Pixels")
plt.ylabel("Pixels")

# Display the figure.
plt.show()


# Find a likely cosmic ray candidate

# cosmic_cluster will store the most likely cosmic ray candidate found.
cosmic_cluster = None

# Track the largest absolute correlation coefficient encountered.
# A large |rho| indicates a strongly asymmetric or elongated detection.
# Cosmic rays often produce very sharp, irregular, or elongated clusters of
# bright pixels that differ from the smoother PSF shape of real stars.
max_rho = 0

# Loop over all detected clusters, including those not accepted as stars.
for cluster in sf_V.clusterList:

    # Covariance matrix of the light distribution for this cluster.
    cov = cluster.cov()

    # The diagonal terms are variances and must be positive.
    # Invalid measurements are skipped.
    if cov[0, 0] <= 0 or cov[1, 1] <= 0:
        continue

    # RMS widths in the x and y directions.
    sigma_x = np.sqrt(cov[0, 0])
    sigma_y = np.sqrt(cov[1, 1])

    # Correlation coefficient derived from the covariance matrix:
    #   rho = cov_xy / (sigma_x * sigma_y)
    # If |rho| is small, the object is more symmetric.
    # If |rho| is large, the object is more distorted, elongated, or skewed.
    rho = cov[0, 1] / (sigma_x * sigma_y)

    # Keep the cluster with the largest absolute rho value.
    #
    # This is a simple way to identify a likely non-stellar object, such as
    # a cosmic ray, because cosmic rays often produce shapes that are much
    # less PSF-like than real stars.
    if abs(rho) > max_rho:
        max_rho = abs(rho)
        cosmic_cluster = cluster


# Plot the image cutout if a candidate was found.
if cosmic_cluster is not None:

    # Get the centroid position of the candidate cluster.
    mux, muy = cosmic_cluster.mu()

    # Convert centroid coordinates to integer array indices.
    x = int(mux)
    y = int(muy)

    # Half-size of the cutout around the candidate.
    #
    # A slightly smaller cutout is used here because cosmic ray events are
    # often confined to a smaller region than stellar PSFs.
    size = 15

    # Extract the local image region around the candidate cluster.
    cutout = sf_V.n[y-size:y+size, x-size:x+size]

    # Create a new figure for the cosmic ray candidate image.
    plt.figure()

    # Display the cutout.
    plt.imshow(cutout, cmap='inferno', origin='lower')

    # Add a colour bar showing the pixel intensity scale.
    plt.colorbar()

    # Axis labels in pixel coordinates.
    plt.xlabel("Pixels")
    plt.ylabel("Pixels")

    # Display the figure.
    plt.show()




