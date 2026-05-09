# Air Quality Forecast App

A Streamlit dashboard for visualising real-time and forecast air quality across Poland using data from the Copernicus Atmosphere Monitoring Service (CAMS).

## Features

- Interactive Folium map with pollutant concentration overlays
- Forecast time-series charts (PM2.5, PM10, NO2, O3) for selected monitoring sites
- Automatic Air Quality Index (AQI) calculation
- CAMS NetCDF data loaded from a GitHub Release asset

## Tech Stack

Python · Streamlit · Plotly · Folium · xarray · rasterio · Pillow · Matplotlib

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

> The app fetches CAMS forecast data on first load. A valid  configuration (CDS API credentials) is required for live data.
