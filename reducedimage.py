# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 21:36:52 2026

@author: nyala
"""


from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

#load calibration frames 
master_bias = fits.getdata(
    r'C:/Users/Nyala/OneDrive/Documents/RHUL/BSc Project/Calibrationdata20260104/n0_master_bias.fits'
)

master_flat_V = fits.getdata(
    r'C:/Users/Nyala/OneDrive/Documents/RHUL/BSc Project/Calibrationdata20260104/f_V_master_flat.fits'
)

#open raw science image (data + header) 
raw_hdu = fits.open(
    r'C:/Users/Nyala/OneDrive/Documents/RHUL/BSc Project/Data_project/NGC752_2026-01-04_17-50-55_V_0035_WCSinfo.fits'
)

Raw_V = raw_hdu[0].data  #the actual pixel values(2D array)
raw_header = raw_hdu[0].header   #the FITS header (keywords + values)
#keep the headers for analysis

#reduce the image
reduced_V = (Raw_V - master_bias) / master_flat_V #applies full calibration equation
#subtracts electronic bias, divides out pixel sensitivity variations
#produces a physically meaningful image

#save reduced image WITH header 
hdu = fits.PrimaryHDU(reduced_V, header=raw_header) #creates FITS file using reduced data, original header. keeps WCS info 
hdu.writeto(
    r'C:/Users/Nyala/OneDrive/Documents/RHUL/BSc Project/Data_project/NGC752_2026-01-04_17-50-55_V_0035_reduced.fits',
    overwrite=True
)

raw_hdu.close() #closes FITS file

# display (stretched)
vmin, vmax = np.percentile(reduced_V, [5, 99]) #limits display, so it ignores extreme pixels. limits to 5% and 99% between the two percentiles 
plt.imshow(reduced_V, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
plt.colorbar()
plt.title("Reduced Raw Image (V) – stretched")
plt.show()

# display (default)
plt.imshow(reduced_V, cmap='gray', origin='lower')
plt.colorbar()
plt.title("Reduced Raw Image (V)")
plt.show()

#to be added to sf.main.py
#comaprison between reduced and raw background model
b_raw = sf_V.b.copy()
imageFileBase_V_red = 'm34_2026-01-04_17-11-20_V_0020_reduced'

sf_V_red = StarField(imageFileDir_V, imageFileBase_V_red)
sf_V_red.setUsableRegion(border, cornerRadius)
sf_V_red.estimateBackground(box, hole, backgroundAlg, useExistingBackground, imageFileDir_V)
b_red = sf_V_red.b.copy()

# Convert both to fractional deviation from their median
b_raw_rel = (b_raw - np.median(b_raw)) / np.median(b_raw)
b_red_rel = (b_red - np.median(b_red)) / np.median(b_red)

# Use one fixed common scale for both
vmin = -0.02
vmax =  0.02

fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
from scipy.ndimage import gaussian_filter

b_raw_plot = gaussian_filter(b_raw_rel, sigma=2)
b_red_plot = gaussian_filter(b_red_rel, sigma=2)

im0 = axes[0].imshow(b_raw_rel, origin='lower', cmap='hot', vmin=vmin, vmax=vmax)
#axes[0].set_title('Background before calibration')
axes[0].set_xlabel('X pixel')
axes[0].set_ylabel('Y pixel')

im1 = axes[1].imshow(b_red_rel, origin='lower', cmap='hot', vmin=vmin, vmax=vmax)
#axes[1].set_title('Background after calibration')
axes[1].set_xlabel('X pixel')
axes[1].set_ylabel('Y pixel')

cbar = fig.colorbar(im1, ax=axes, shrink=0.9)
cbar.set_label('Fractional deviation from median background')

plt.show()

# Optional quantitative check
print("Raw fractional std     =", np.std(b_raw_rel))
print("Reduced fractional std =", np.std(b_red_rel))

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
plt.figure()
plt.hist(pixels, bins=300, histtype='step', color='red')
plt.yscale('log')
plt.xlabel('Pixel counts (ADU)')
plt.ylabel('Number of pixels')
plt.title(f'Pixel count histogram ({2*halfBox+1}×{2*halfBox+1} box at ({center_row},{center_col}))')
plt.show()
