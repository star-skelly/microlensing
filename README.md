# Microlensing Research Project
This project essentially generates synthetic microlensing data to use machine learning to characterize events and identify exoplanets.

https://github.com/user-attachments/assets/663a01cc-63db-4e69-941e-a0504090d89e

Microlensing events occur when two stars line up with each other and the observer. The star in the middle acts as a lens, magnifying the light from the source star behind it. However, if the lens star has a planet orbiting it, that shows up in the light curve of the event! So, with machine learning, this project aims to characterize these events and extract the possible planetary parameters from them. 

To do this, we generate high Signal-to-Noise ratio synthetic microlensing data and store the ground truth parameter values for these curves. This allows us to test different machine learning techniques and measure their efficacy before applying them to real world data.

## Conclusions

- Some variables (like alpha, the angle between source trajectory + binary axis) are nearly impossible to derive simply from the data
- Simple physics-informed models like fitting the known (but complicated) microlensing still outperform new deep-learning  models (given current data, this project is in hiatus)
- Machine Learning models tested (with various data massaging) include: CNNs, MLPs, Transformers, simple Least-Squares Curve Fitting

## Setup
``` pip install -r requirements.txt ```
Then run the script provided to pull microlensing events from NASA sources, and run the data_generation_pipeline.

## How does it work?
- **data_generation_pipeline.ipynb** generates synthetic microlensing events with real lightcurve noise injected
- **visualize_params.ipynb** has a lovely visualization of the microlensing simulation parameters, allowing for deeper understanding of microlensing events with and without planets
- **mulens_ex.ipynb** has some of the initial loading and playing around with lightcurves and microlensing simulations
