# Star Cluster Photometry (BSc Project)

This repository contains the Python code used for photometry and isochrone fitting of the open clusters M34, M35 and M37.

## Overview

The analysis includes:
- Star detection using a custom Starfinder-based algorithm
- Parameter optimisation for reliable source extraction
- CCD calibration (bias subtraction and flat-field correction)
- Photometric calibration to the Johnson B and V system
- Isochrone fitting using MIST stellar evolution models

## Files

- `ParameterOpt.py` – star detection and parameter optimisation  
- `masterbias.py` – creation of master bias frame  
- `masterflatB_V.py` – flat-field calibration  
- `reducedimage.py` – application of calibration to images  
- `CalibrationBV.py` / `CalibrationAdvanced.py` – photometric calibration  
- `fit_isochrone.py` – isochrone fitting and CMD analysis  

## Acknowledgements

Parts of the Starfinder code (`sf.main.py`, `sf.lib.py`) were provided by Professor Glen Cowan and adapted for this project.
