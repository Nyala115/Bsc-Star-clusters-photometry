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
