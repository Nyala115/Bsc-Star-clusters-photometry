# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 21:17:01 2026

@author: nyala
"""
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

# Load master bias
master_bias = fits.getdata(
    r'C:/Users/Nyala/OneDrive/Documents/RHUL/BSc Project/Calibrationdata20260104/n0_master_bias.fits'
)

# B filter flats 
flat_B_files = [
    r'C:/Users/Nyala/OneDrive/Documents/RHUL/BSc Project/Calibrationdata20260104/flat_2026-01-04_16-22-06_B_0005.fits',
    r'C:/Users/Nyala/OneDrive/Documents/RHUL/BSc Project/Calibrationdata20260104/flat_2026-01-04_16-23-07_B_0010.fits',
    r'C:/Users/Nyala/OneDrive/Documents/RHUL/BSc Project/Calibrationdata20260104/flat_2026-01-04_16-23-12_B_0011.fits',
    r'C:/Users/Nyala/OneDrive/Documents/RHUL/BSc Project/Calibrationdata20260104/flat_2026-01-04_16-23-17_B_0012.fits'
]

# V filter flats 
flat_V_files = [
    r'C:/Users/Nyala/OneDrive/Documents/RHUL/BSc Project/Calibrationdata20260104/flat_2026-01-04_16-22-31_V_0006.fits',
    r'C:/Users/Nyala/OneDrive/Documents/RHUL/BSc Project/Calibrationdata20260104/flat_2026-01-04_16-22-49_V_0007.fits',
    r'C:/Users/Nyala/OneDrive/Documents/RHUL/BSc Project/Calibrationdata20260104/flat_2026-01-04_16-22-54_V_0008.fits',
    r'C:/Users/Nyala/OneDrive/Documents/RHUL/BSc Project/Calibrationdata20260104/flat_2026-01-04_16-22-58_V_0009.fits'
]


def create_master_flat(flat_files, master_bias):

    # Load flats and subtract bias
    flats = [(fits.getdata(f) - master_bias) for f in flat_files]

    # Stack into a 3D array so images can be combined pixel-by-pixel
    flat_stack = np.array(flats)

    # Median combine the flats
    median_flat = np.median(flat_stack, axis=0)

    # Compute median of the entire image
    norm_factor = np.median(median_flat)

    # Normalize the flat so the median pixel value becomes 1
    master_flat = median_flat / norm_factor

    return master_flat


# Create master flats
master_flat_B = create_master_flat(flat_B_files, master_bias)
master_flat_V = create_master_flat(flat_V_files, master_bias)



# DISPLAY V-FILTER MASTER FLAT WITH BETTER CONTRAST

median_V = np.median(master_flat_V)

# Clip colour scale to ±5% around median
vmin_V = 0.95 * median_V
vmax_V = 1.05 * median_V

plt.figure()

plt.imshow(master_flat_V,
           cmap='hot',
           origin='lower',
           vmin=vmin_V,
           vmax=vmax_V)

plt.colorbar()

#plt.title("Master Flat – V filter")

plt.show()



# DISPLAY B-FILTER MASTER FLAT WITH BETTER CONTRAST


median_B = np.median(master_flat_V)

# Clip colour scale to ±5% around median
vmin_B = 0.95 * median_B
vmax_B = 1.05 * median_B

plt.figure()

plt.imshow(master_flat_B,
           cmap='hot',
           origin='lower',
           vmin=vmin_B,
           vmax=vmax_B)

plt.colorbar()

#plt.title("Master Flat – B filter")

plt.show()

# SAVE MASTER FLATS

fits.writeto(
    r'C:/Users/Nyala/OneDrive/Documents/RHUL/BSc Project/Calibrationdata20260104/f_B_master_flat.fits',
    master_flat_B,
    overwrite=True
)

fits.writeto(
    r'C:/Users/Nyala/OneDrive/Documents/RHUL/BSc Project/Calibrationdata20260104/f_V_master_flat.fits',
    master_flat_V,
    overwrite=True
)


# ZOOMED VIEW OF A DUST DONUT


# Choose the centre of the donut (adjust these numbers)
x_center = 3900
y_center = 2000

# Size of the zoom window in pixels
zoom_size = 300

# Extract a small region around the donut
zoom_region = master_flat_V[
    y_center - zoom_size : y_center + zoom_size,
    x_center - zoom_size : x_center + zoom_size
]

# Compute display limits relative to the median
median_zoom = np.median(zoom_region)

vmin = 0.97 * median_zoom
vmax = 1.03 * median_zoom


plt.figure(figsize=(5,5))

plt.imshow(zoom_region,
           cmap='hot',
           origin='lower',
           vmin=vmin,
           vmax=vmax)

plt.colorbar()

#plt.title("Dust Donut Feature (Zoomed Flat Field)")

plt.show()