# Migration Estimation

Machine-learning pipeline that estimates disaster-displaced populations for
rapid-onset events (floods and cyclones) from satellite-derived and
socioeconomic features.

## Overview

The pipeline builds an area-of-interest for each disaster event, extracts
features from Google Earth Engine, joins socioeconomic indicators, and trains
regression models to predict the number of displaced persons.

- **Satellite features (GEE):** SAR flood extent, population exposed,
  nighttime-light outages, precipitation, wind, slope, TWI, built surface.
- **Socioeconomic features:** infant mortality rate, HDI, relative wealth index.
- **Models:** Ridge baseline, XGBoost (Optuna-tuned), Random Forest, with SHAP
  explanations and grouped cross-validation.

## Project structure

```
notebook/
  notebook.ipynb   # main pipeline notebook
  notebook.py      # same pipeline as a script
```

## Requirements

Python 3 with the packages installed at the top of the notebook:

```bash
pip install xgboost shap scikit-learn geopandas pyarrow geemap \
  earthengine-api optuna optuna-integration cartopy rasterio \
  statsmodels pycountry pycountry-convert
```

A Google Earth Engine account is required for feature extraction.

## Usage

Open `notebook/notebook.ipynb` (designed for Google Colab with Drive-mounted
data) and run the cells top to bottom. Adjust paths and parameters in the
`Config` dataclass to match your environment.

## License

[MIT](LICENSE)
