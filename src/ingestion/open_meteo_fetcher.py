# Open-Meteo Historical Weather Fetcher
# Dataset: ERA5 Reanalysis Daily Precipitation
# Source: Open-Meteo (Free, No Auth needed)
# URL: https://archive-api.open-meteo.com/v1/archive

import os
import requests
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from pathlib import Path

class OpenMeteoFetcher:
    """
    Fetches historical daily rainfall data from the Open-Meteo Archive API.
    
    To avoid rate limits on massive ocean grids, this pulls a sparse 1.0 degree 
    grid and relies on the UnifiedLoader's xarray interpolation to seamlessly 
    align it with the high-resolution Copernicus marine grid.
    
    Used as Seasonal Factor (S) proxy in RM-NPI:
        High rainfall during monsoon -> S = 1
        Low rainfall during dry season -> S = 0
    """

    def __init__(self, output_dir: str = "data/raw/openmeteo"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"

    def fetch(
        self,
        start_date: str,
        end_date: str,
        lat_min: float = 0.0,
        lat_max: float = 25.0,
        lon_min: float = 65.0,
        lon_max: float = 90.0,
    ) -> xr.Dataset:
        """
        Fetch daily rainfall from Open-Meteo, aggregate to monthly, and convert to xarray.
        """
        print(f"[Open-Meteo] Fetching historical rainfall: {start_date} to {end_date}")
        print(f"             Region: lat[{lat_min},{lat_max}], lon[{lon_min},{lon_max}]")

        # 1. Create a sparse grid (1.0 degree resolution) for API efficiency
        resolution = 1.0
        lats = np.arange(lat_min, lat_max + resolution, resolution)
        lons = np.arange(lon_min, lon_max + resolution, resolution)
        grid_points = [(lat, lon) for lat in lats for lon in lons]
        
        all_data = []
        chunk_size = 90 # Open-Meteo allows max 100 locations per API call
        
        # 2. Query the bulk API in chunks
        for i in range(0, len(grid_points), chunk_size):
            chunk = grid_points[i:i+chunk_size]
            chunk_lats = [p[0] for p in chunk]
            chunk_lons = [p[1] for p in chunk]
            
            params = {
                "latitude": chunk_lats,
                "longitude": chunk_lons,
                "start_date": start_date,
                "end_date": end_date,
                "daily": "precipitation_sum",
                "timezone": "GMT"
            }
            
            try:
                response = requests.get(self.base_url, params=params, timeout=60)
                response.raise_for_status()
                data = response.json()
                
                # Format: List of dicts if multiple locations
                if isinstance(data, dict) and "daily" in data:
                    data = [data]
                    
                for j, loc_data in enumerate(data):
                    if "daily" in loc_data and "precipitation_sum" in loc_data["daily"]:
                        df = pd.DataFrame(loc_data["daily"])
                        df["lat"] = chunk_lats[j]
                        df["lon"] = chunk_lons[j]
                        all_data.append(df)
                        
            except Exception as e:
                print(f"  [!] API Chunk Failed: {e}")
                continue
                
        if not all_data:
            raise ValueError("No rainfall data was successfully fetched from Open-Meteo")
            
        # 3. Combine Data and Aggregate Daily to Monthly Sums
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df["time"] = pd.to_datetime(combined_df["time"])
        
        # Aggregate to monthly sum (e.g. Total June Rainfall)
        combined_df['year_month'] = combined_df['time'].dt.to_period('M')
        monthly_df = combined_df.groupby(['year_month', 'lat', 'lon'], as_index=False).agg({
            'precipitation_sum': 'sum'
        })
        # Reset the timestamp to the first of the month to easily align with Copernicus
        monthly_df['time'] = monthly_df['year_month'].dt.to_timestamp()
        monthly_df = monthly_df.drop(columns=['year_month'])
        
        # 4. Save cache
        filename = f"openmeteo_{start_date}_{end_date}.csv"
        csv_path = self.output_dir / filename
        monthly_df.to_csv(csv_path, index=False)
        print(f"             Aggregated {len(combined_df):,} daily records into {len(monthly_df):,} monthly data points.")
        print(f"             Cached to: {csv_path}")

        # 5. Convert to an xarray Dataset geometry (matching CHIRPS format for UnifiedLoader)
        monthly_df = monthly_df.set_index(['time', 'lat', 'lon'])
        ds = monthly_df.to_xarray()
        
        n_time = ds.dims.get("time", 0)
        n_lat = ds.dims.get("lat", 0)
        n_lon = ds.dims.get("lon", 0)
        print(f"[Open-Meteo] Combined Geometry: time={n_time}, lat={n_lat}, lon={n_lon}")
        
        return ds

    def fetch_latest(
        self,
        n_months: int = 3,
        lat_min: float = 0.0,
        lat_max: float = 25.0,
        lon_min: float = 65.0,
        lon_max: float = 90.0,
    ) -> xr.Dataset:
        end = datetime.utcnow()
        start = end - timedelta(days=n_months * 30)
        return self.fetch(
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
            lat_min, lat_max, lon_min, lon_max,
        )
