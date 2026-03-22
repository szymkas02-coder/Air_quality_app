import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xarray as xr
import folium
from streamlit_folium import st_folium
import base64
import os
import io
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from PIL import Image
import rasterio
import json
from rasterio.warp import calculate_default_transform, reproject, Resampling
from datetime import datetime, timedelta
from branca.element import MacroElement
from jinja2 import Template

# Import the CAMS reading functionality
from cams_read import get_cams_air_quality, get_latest_forecast_meta

# Set page configuration
st.set_page_config(
    page_title="Air Quality Forecast App",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .forecast-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .data-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Sample site data (you can modify these coordinates)
with open("sample_sites_poland.json", "r", encoding="utf-8") as plik:
    SAMPLE_SITES = json.load(plik)

def load_cams_data():
    return get_cams_air_quality()

def get_air_quality_index(pm25, pm10, no2, o3):
    """Calculate Air Quality Index based on pollutant concentrations"""
    # Simplified AQI calculation (you can implement more sophisticated methods)
    aqi = 0
    
    # PM2.5 contribution
    if pm25 <= 12:
        aqi = max(aqi, 50 * (pm25 / 12))
    elif pm25 <= 35.4:
        aqi = max(aqi, 51 + 49 * ((pm25 - 12) / (35.4 - 12)))
    elif pm25 <= 55.4:
        aqi = max(aqi, 101 + 49 * ((pm25 - 35.4) / (55.4 - 35.4)))
    elif pm25 <= 150.4:
        aqi = max(aqi, 151 + 49 * ((pm25 - 55.4) / (150.4 - 55.4)))
    elif pm25 <= 250.4:
        aqi = max(aqi, 201 + 49 * ((pm25 - 150.4) / (250.4 - 150.4)))
    else:
        aqi = max(aqi, 251 + 49 * ((pm25 - 250.4) / (500.4 - 250.4)))
    
    # PM10 contribution
    if pm10 <= 54:
        aqi = max(aqi, 50 * (pm10 / 54))
    elif pm10 <= 154:
        aqi = max(aqi, 51 + 49 * ((pm10 - 54) / (154 - 54)))
    elif pm10 <= 254:
        aqi = max(aqi, 101 + 49 * ((pm10 - 154) / (254 - 154)))
    elif pm10 <= 354:
        aqi = max(aqi, 151 + 49 * ((pm10 - 254) / (354 - 254)))
    elif pm10 <= 424:
        aqi = max(aqi, 201 + 49 * ((pm10 - 354) / (424 - 354)))
    else:
        aqi = max(aqi, 251 + 49 * ((pm10 - 424) / (604 - 424)))
    
    return min(500, int(aqi))

def get_aqi_category(aqi):
    """Get AQI category and color"""
    if aqi <= 50:
        return "Good", "green", "🟢"
    elif aqi <= 100:
        return "Moderate", "orange", "🟡"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "red", "🟠"
    elif aqi <= 200:
        return "Unhealthy", "purple", "🔴"
    elif aqi <= 300:
        return "Very Unhealthy", "maroon", "🟣"
    else:
        return "Hazardous", "black", "⚫"

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
import folium

def reproject_to_webmercator(lons, lats, values):
    """Reproject a lat/lon grid to Web Mercator"""

    transform_in = rasterio.transform.from_bounds(
        west=lons.min(), south=lats.min(), east=lons.max(), north=lats.max(),
        width=values.shape[1], height=values.shape[0]
    )

    src_crs = "EPSG:4326"
    dst_crs = "EPSG:3857"

    transform, width, height = calculate_default_transform(
        src_crs, dst_crs, values.shape[1], values.shape[0],
        lons.min(), lats.min(), lons.max(), lats.max()
    )

    dst = np.empty((height, width), dtype=values.dtype)

    reproject(
        source=values,
        destination=dst,
        src_transform=transform_in,
        src_crs=src_crs,
        dst_transform=transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest
    )

    return dst, lats, lons

def create_air_quality_map(ds, selected_pollutant, selected_step, loc=[51.0, 17.0]):
    if selected_pollutant not in ds.data_vars:
        return None
    if 'time' not in ds.dims:
        return None

    data = ds[selected_pollutant].sel(time=selected_step, method='nearest')

    lats = data.latitude.values
    lons = data.longitude.values
    values = data.values.astype(float)

    # Reproject to Web Mercator
    values_proj, lats_fixed, lons_fixed = reproject_to_webmercator(lons, lats, values)

    # Normalize and colormap
    vmin = np.nanmin(values_proj)
    vmax = np.nanmax(values_proj)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.get_cmap('YlOrRd')

    rgba_img = cmap(norm(values_proj))
    rgba_img_uint8 = (rgba_img * 255).astype(np.uint8)
    img = Image.fromarray(rgba_img_uint8, mode='RGBA')

    # Save image to memory (PNG)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    # Encode to base64
    encoded = base64.b64encode(img_byte_arr.read()).decode('utf-8')
    img_data = f"data:image/png;base64,{encoded}"

    # Bounds in EPSG:4326
    bounds = [
        [float(lats_fixed.min()), float(lons_fixed.min())],
        [float(lats_fixed.max()), float(lons_fixed.max())]
    ]

    # Create map
    m = folium.Map(location=loc, zoom_start=7, tiles='OpenStreetMap')

    folium.raster_layers.ImageOverlay(
        image=img_data,
        bounds=bounds,
        opacity=0.7,
        interactive=True,
        cross_origin=False,
        zindex=1,
    ).add_to(m)

    # --- Create and add colorbar ---
    def create_colorbar(cmap, vmin, vmax, label):
        fig, ax = plt.subplots(figsize=(4, 0.4))
        fig.subplots_adjust(bottom=0.5)

        cb = plt.colorbar(
            plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
            cax=ax,
            orientation='horizontal'
        )
        cb.set_label(label, fontsize=8)
        cb.ax.tick_params(labelsize=8)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', transparent=True)
        plt.close(fig)
        buf.seek(0)

        b64 = base64.b64encode(buf.read()).decode('utf-8')
        html = f'<img src="data:image/png;base64,{b64}" style="position: fixed; bottom: 10px; left: 10px; width: 250px; z-index: 9999;">'

        class FloatImage(MacroElement):
            def __init__(self, html):
                super().__init__()
                self._template = Template(f"""
                    {{% macro script(this, kwargs) %}}
                    var img = `{html}`;
                    var div = L.DomUtil.create('div');
                    div.innerHTML = img;
                    document.body.appendChild(div);
                    {{% endmacro %}}
                """)

        return FloatImage(html)

    label = f"{selected_pollutant} [µg/m³]"
    colorbar = create_colorbar(cmap, vmin, vmax, label)
    m.add_child(colorbar)

    return m

def get_site_forecast(ds, site_lat, site_lon, pollutant):
    """Get forecast for a specific site"""
    try:
        # Find nearest grid point
        lat_idx = np.abs(ds.latitude.values - site_lat).argmin()
        lon_idx = np.abs(ds.longitude.values - site_lon).argmin()
        
        # Extract time series for the site
        if pollutant in ds.data_vars:
            site_data = ds[pollutant].isel(latitude=lat_idx, longitude=lon_idx)
            
            # Handle different coordinate systems
            if 'time' in ds.dims:
                # Use time if available
                df = site_data.to_dataframe().reset_index()
                df = df[['time', pollutant]]
                df.columns = ['time', 'value']
                df['time'] = pd.to_datetime(df['time'])

            return df
        
    except Exception as e:
        st.error(f"Error getting forecast for site: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🌤️ Air Quality Forecast Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("Settings")
    
    # Data management section
    st.sidebar.subheader("📊 Data Management")

    # Przycisk odświeżania danych
    refresh_data = st.sidebar.button("🔄 Refresh CAMS Data")
   
    if refresh_data:
        st.cache_resource.clear()
    
    # Load data, always using force_download from session state
    (ds, data_date) = load_cams_data()
    
    if ds is None:
        st.error("Failed to load air quality data")
        return
    
    # Display data information
    if data_date:
        st.markdown(f"""
        <div class="data-info">
            <h4>📅 Data Information</h4>
            <p><strong>Current Data Date:</strong> {data_date}</p>
            <p><strong>Data Source:</strong> CAMS Europe Air Quality Forecasts</p>
            <p><strong>Coverage:</strong> 96-hour forecast</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.subheader("Map Settings")
    
    # Pollutant selection with nicer names and units
    pollutants = [
        var for var in ds.data_vars
        if var not in ['latitude', 'longitude', 'time', 'step']
    ]

    # Build display names: "PM2.5 (µg/m3)", "Ozone (µg/m3)", "Birch Pollen (grains/m3)", etc.
    display_names = []
    pollutant_map = {}
    pm10_index = 0
    for i, var in enumerate(pollutants):
        attrs = ds[var].attrs
        # Use 'species' if available, fallback to variable name
        species = attrs.get('species', var.replace('_', ' ').title())
        # Rename particulate matter to PM
        if var == 'particulate_matter_2.5um':
            species = 'PM2.5'
        elif var == 'particulate_matter_10um':
            species = 'PM10'
            pm10_index = i
        # Remove "Aerosol" from PM names
        species = species.replace('Aerosol', '').strip()
        # Remove "Grain" from pollen
        species = species.replace('Grain', '').strip()
        units = attrs.get('units', '')
        display = f"{species} ({units})" if units else species
        display_names.append(display)
        pollutant_map[display] = var

    # Default to PM10 if available
    selected_display = st.sidebar.selectbox(
        "Select Pollutant",
        display_names,
        index=pm10_index if 'PM10' in display_names[pm10_index] else 0
    )
    selected_pollutant = pollutant_map[selected_display]
    
    if 'time' in ds.dims:
        available_times = pd.to_datetime(ds.time.values)
        # Find the index closest to the current UTC hour
        now = datetime.utcnow()
        time_diffs = [abs((t - now).total_seconds()) for t in available_times]
        default_time_index = int(np.argmin(time_diffs))
        selected_step = st.sidebar.selectbox(
            "Select Forecast Time",
            available_times,
            index=default_time_index
        )
    selected_time_index = int(np.where(available_times == selected_step)[0][0])
    
    st.sidebar.subheader("Site Selection")
    search_term = st.sidebar.text_input("Search site:")

    filtered_sites = [name for name in SAMPLE_SITES if search_term.lower() in name.lower()]
    if not search_term:
        filtered_sites = list(SAMPLE_SITES.keys())

    # Domyślnie Warszawa
    default_city = "Warszawa"
    default_index = filtered_sites.index(default_city) if default_city in filtered_sites else 0

    selected_site = st.sidebar.selectbox(
        "Select Site for Detailed Forecast",
        filtered_sites,
        index=default_index
    )

    st.write(f"Selected: {selected_site} - {SAMPLE_SITES[selected_site]}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🌍 Air Quality Map")
        
        # Create and display the map
        coords = [SAMPLE_SITES[selected_site]["lat"], SAMPLE_SITES[selected_site]["lon"]]
        air_quality_map = create_air_quality_map(ds, selected_pollutant, selected_step, coords)
        if air_quality_map:
            # Add marker for selected city
            folium.Marker(
                location=coords,
                popup=f"{selected_site}",
                icon=folium.Icon(color="blue")
            ).add_to(air_quality_map)
            # Display map and handle click events
            map_data = st_folium(air_quality_map, width=700, height=500)

        else:
            st.warning("Unable to create map for selected pollutant/time")
    
    with col2:
        st.subheader("📊 Current conditions")

        # Get current site info
        site_info = SAMPLE_SITES[selected_site]
        site_lat = site_info['lat']
        site_lon = site_info['lon']

        # Display site info
        st.markdown(f"""
        <div class="metric-card">
            <h4>{selected_site}</h4>
            <p>📍 {site_lat:.4f}, {site_lon:.4f}</p>
        </div>
        """, unsafe_allow_html=True)

        # Get current values for key pollutants at selected site (nearest grid point)
        current_data = {}
        key_pollutants = {
            'particulate_matter_2.5um': 'PM2.5',
            'particulate_matter_10um': 'PM10',
            'nitrogen_dioxide': 'NO₂',
            'ozone': 'O₃'
        }

        # Find nearest grid point indices
        lat_idx = np.abs(ds.latitude.values - site_lat).argmin()
        lon_idx = np.abs(ds.longitude.values - site_lon).argmin()

        for pollutant, label in key_pollutants.items():
            if pollutant in ds.data_vars:
                try:
                    current_val = ds[pollutant].isel(time=selected_time_index, latitude=lat_idx, longitude=lon_idx).values
                    if not np.isnan(current_val):
                        current_data[label] = float(current_val)
                except Exception:
                    pass

        # Display current values
        if current_data:
            for label, value in current_data.items():
                st.metric(label, f"{value:.2f}")

        # Calculate and display AQI if all required pollutants are available
        pm25 = current_data.get('PM2.5', np.nan)
        pm10 = current_data.get('PM10', np.nan)
        no2 = current_data.get('NO₂', np.nan)
        o3 = current_data.get('O₃', np.nan)
        if not any(np.isnan(x) for x in [pm25, pm10, no2, o3]):
            aqi = get_air_quality_index(pm25, pm10, no2, o3)
            aqi_cat, aqi_color, aqi_emoji = get_aqi_category(aqi)
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid {aqi_color};">
                <h4>Air Quality Index {aqi_emoji}</h4>
                <p style="font-size:2rem; color:{aqi_color};"><b>{aqi}</b> - {aqi_cat}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Detailed forecast section
    st.subheader("📈 Detailed Forecast")
    
    # Get forecast for selected site
    forecast_data = get_site_forecast(ds, site_info['lat'], site_info['lon'], selected_pollutant)
    
    if forecast_data is not None:
        # Create forecast plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=forecast_data['time'],
            y=forecast_data['value'],
            mode='lines+markers',
            name=f'{selected_pollutant.replace("_", " ").title()}',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=6)
        ))
        
        # Get units and species from dataset attributes if available
        units = ds[selected_pollutant].attrs.get('units', '')
        species = ds[selected_pollutant].attrs.get('species', selected_pollutant.replace('_', ' ').title())

        fig.update_layout(
            title=f"97-Step Forecast for {selected_site} - {species}",
            xaxis_title="Time",
            yaxis_title=f"Concentration ({units})" if units else "Concentration",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display forecast table
        st.subheader("📋 Forecast Data Table")
        forecast_display = forecast_data.copy()
        
        # Handle different column structures safely
        if len(forecast_display.columns) == 2:
            if 'time' in forecast_display.columns:
                forecast_display['time'] = forecast_display['time'].dt.strftime('%Y-%m-%d %H:%M')
                forecast_display.columns = ['Time', 'Value']
            elif 'step' in forecast_display.columns:
                forecast_display.columns = ['Step', 'Value']
            else:
                forecast_display.columns = ['X', 'Value']
        elif len(forecast_display.columns) == 3:
            # Keep original column names if we have 3 columns
            pass
        
        st.dataframe(forecast_display, use_container_width=True)
    
    # Data storage information
    st.sidebar.subheader("💾 Data Info")
    meta = get_latest_forecast_meta()
    if meta:
        st.sidebar.info(f"📅 Forecast: {meta['date']}\n⏱ Updated: {meta['downloaded_at_utc'][:16]} UTC")

    # Additional information
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.info("""
    This app provides air quality forecasts using **CAMS (Copernicus Atmosphere Monitoring Service) European Air Quality Forecasts**.

    **Data Source**: CAMS Europe Air Quality Forecasts  
    **Update Frequency**: Daily (automatic)  
    **Coverage**: Europe  
    **Smart Caching**: Only downloads new data when needed

    **Licence**:  
    The CAMS data are provided under the [Copernicus Licence](https://apps.ecmwf.int/datasets/licences/copernicus/), which allows free and open access to Copernicus information and data.
    """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🌤️ Air Quality Forecast Dashboard | Powered by CAMS & Streamlit</p>
        <p>📅 Smart caching system with date-based file management</p>
        <p>Data source: CAMS European Air Quality Forecasts &mdash; <a href='https://atmosphere.copernicus.eu/'>Copernicus Atmosphere Monitoring Service</a></p>
        <p>Licence: <a href='https://apps.ecmwf.int/datasets/licences/copernicus/'>Copernicus Licence</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
