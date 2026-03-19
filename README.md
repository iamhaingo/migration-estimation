# Climate-Related Human Displacement Estimation

A deep learning framework for estimating internal displacement caused by climate-related events using satellite imagery.

## Overview

This repository provides a multi-branch convolutional neural network that fuses:
- **Landsat 9** RGB imagery (30m resolution, 10×10 km tiles)
- **NASA Black Marble** VIIRS nighttime lights (pre/post-event difference composites)

to predict the number of Internally Displaced Persons (IDPs) following a climate event.

Two task formulations are supported:
- **Regression** — predict raw IDP counts 
- **Ordinal classification** — categorise events into Minimal / Moderate / Severe tiers

The regression head outputs a scalar IDP count. The ordinal classification head uses a [CORAL](https://doi.org/10.1016/j.patrec.2020.11.008) layer to preserve severity ordering.