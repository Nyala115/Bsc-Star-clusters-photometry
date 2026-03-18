# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 17:38:15 2026

@author: nyala
"""

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt


bias_files = [ #creates a python list
    r'c:\users\nyala\onedrive\documents\rhul\bsc project\Calibrationdata20260104\bias_-20C_0013.fits',
    r'c:\users\nyala\onedrive\documents\rhul\bsc project\Calibrationdata20260104\bias_-20C_0014.fits',
    r'c:\users\nyala\onedrive\documents\rhul\bsc project\Calibrationdata20260104\bias_-20C_0015.fits',  
    r'c:\users\nyala\onedrive\documents\rhul\bsc project\Calibrationdata20260104\bias_-20C_0016.fits'
]

# Load all bias frames into a 3D array
bias_images = [fits.getdata(f) for f in bias_files] #reads FITS file extracts image data stores as a numpy array
bias_stack = np.array(bias_images) #converts list into a single 3D NumPy array 
#(3D array enforces structure of treating each pixel independently and combining only corresponding pixels across frames)

# Compute the median value at each pixel position across all bias frames. Median is independent of outliers
master_bias = np.median(bias_stack, axis=0)  #it says for each pixels take median across all bias frames

# Look at the result (check for issues)

plt.imshow(master_bias, cmap='gray', origin='lower', )
plt.colorbar()
plt.title("Master Bias Frame")
plt.show()

# Save the master bias as a new FITS file
import os
hdu = fits.PrimaryHDU(master_bias)
hdu.writeto(r'c:\users\nyala\onedrive\documents\rhul\bsc project\Calibrationdata20260104\n0_master_bias.fits', overwrite=True)



print(os.getcwd()) #check where its saved

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os


# LIST OF BIAS FILES
bias_files = [
    r'c:\users\nyala\onedrive\documents\rhul\bsc project\Calibrationdata20260104\bias_-20C_0013.fits',
    r'c:\users\nyala\onedrive\documents\rhul\bsc project\Calibrationdata20260104\bias_-20C_0014.fits',
    r'c:\users\nyala\onedrive\documents\rhul\bsc project\Calibrationdata20260104\bias_-20C_0015.fits',
    r'c:\users\nyala\onedrive\documents\rhul\bsc project\Calibrationdata20260104\bias_-20C_0016.fits'
]
# This list stores the file paths of the individual bias frames.



# LOAD BIAS FRAMES
bias_images = [fits.getdata(f) for f in bias_files]
# Read each FITS file and extract the image data.
# Each image becomes a 2D NumPy array.

bias_stack = np.array(bias_images)
# Convert the list of images into a 3D NumPy array.
# Shape becomes (number_of_frames, image_height, image_width).



# CREATE MASTER BIAS FRAME
master_bias = np.median(bias_stack, axis=0)
# Compute the median value for each pixel position across all bias frames.
# Using the median reduces the influence of outliers or hot pixels.



# DISPLAY MASTER BIAS IMAGE
plt.figure()

plt.imshow(master_bias, cmap='gray', origin='lower')
# Show the master bias frame as an image.
# origin='lower' ensures astronomical orientation.

plt.colorbar()
# Adds a colour bar showing pixel count values.

#plt.title("Master Bias Frame")

plt.show()



# HISTOGRAM OF PIXEL VALUES
pixels = master_bias.flatten()
# Convert the 2D image into a 1D array containing every pixel value.

plt.figure()

plt.hist(pixels, bins=300, histtype='step')
# Plot a histogram of the pixel values.
# bins=300 gives good resolution for CCD noise distributions.

plt.yscale('log')
# Log scale helps visualise the distribution tails more clearly.

plt.xlabel("Pixel Counts (ADU)")
plt.ylabel("Number of Pixels")

plt.title("Histogram of Master Bias Pixel Values")

plt.show()



# SAVE MASTER BIAS
hdu = fits.PrimaryHDU(master_bias)
# Create a new FITS file containing the master bias frame.

hdu.writeto(
    r'c:\users\nyala\onedrive\documents\rhul\bsc project\Calibrationdata20260104\n0_master_bias.fits',
    overwrite=True
)
# Save the master bias file.


print(os.getcwd())
# Print the current working directory (useful for debugging file paths).
