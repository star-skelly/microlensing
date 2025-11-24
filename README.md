# Microlensing Research Project
This project essentially generates synthetic microlensing data to use machine learning to characterize events and identify exoplanets.

## Setup
``` pip install -r requirements.txt ```
Then run the script provided to pull microlensing events from NASA sources, and run the data_generation_pipeline.

## How does it work?
Still in very early stages, but the concept builds off of the idea of synthetic data to train models, which uses a teacher to distill into a student model that works with real world data.

- **data_generation_pipeline.ipynb** generates synthetic microlensing events with real lightcurve noise injected
- **visualize_params.ipynb** has a lovely visualization of the microlensing simulation parameters, allowing for deeper understanding
- **mulens_ex.ipynb** has some of the initial loading and playing around with lightcurves and microlensing simulations
