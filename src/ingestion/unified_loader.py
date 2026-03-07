# Unified Data Loader -- Copernicus as primary source, CHIRPS as rainfall supplement
import os
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from .copernicus_fetcher import CopernicusFetcher
from .chirps_fetcher import CHIRPSFetcher

load_dotenv()


class UnifiedLoader:
    """
    Orchestrates data fetching with Copernicus as primary source.

    Primary: Copernicus Marine (physics + BGC)
        thetao, so, uo, vo, zos -- physics
        no3, po4, o2, chl       -- biogeochemistry

    Supplement: CHIRPS Rainfall (public, no auth)
        precip -> seasonal_factor (S component of RM-NPI)

    All sources merged into a single flat DataFrame for the autoencoder.
    """

    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.copernicus = CopernicusFetcher()
        self.chirps = CHIRPSFetcher()

    def fetch_all(
        self,
        start_date: str,
        end_date: str,
        lat_min: float = 5.0,
        lat_max: float = 20.0,
        lon_min: float = 70.0,
        lon_max: float = 85.0,
        skip_on_error: bool = True,
    ) -> dict:
        """
        Fetch from all sources. Copernicus is the primary; CHIRPS is supplement.
        Returns dict of xarray Datasets.
        """
        datasets = {}

        # Primary: Copernicus (needs credentials)
        print("\n" + "=" * 55)
        print("SOURCE 1/2: Copernicus Marine (Physics + BGC)")
        print("  Variables: thetao, so, uo, vo, zos, no3, po4, o2, chl")
        print("=" * 55)
        try:
            datasets["copernicus"] = self.copernicus.fetch(
                start_date, end_date,
                lat_min, lat_max, lon_min, lon_max,
            )
            print("[OK] Copernicus fetched successfully")
        except Exception as e:
            print(f"  [!] Copernicus fetch failed: {e}")
            if not skip_on_error:
                raise

        # Supplement: CHIRPS rainfall (public, no auth)
        print("\n" + "=" * 55)
        print("SOURCE 2/2: CHIRPS Rainfall (Seasonal Factor)")
        print("=" * 55)
        try:
            datasets["chirps_rain"] = self.chirps.fetch(
                start_date, end_date,
                lat_min, lat_max, lon_min, lon_max,
            )
        except Exception as e:
            print(f"  [!] CHIRPS fetch failed: {e}")
            if not skip_on_error:
                raise

        n_ok = len(datasets)
        print(f"\n[OK] Fetched {n_ok}/2 data sources")
        return datasets

    def align_and_merge(
        self,
        datasets: dict,
        target_resolution: float = 0.25,
    ) -> pd.DataFrame:
        """
        Flatten and merge all datasets into a rich single DataFrame.

        Steps:
            1. Copernicus -> to_dataframe() (already has computed columns)
            2. Regrid Copernicus to target_resolution
            3. CHIRPS -> seasonal_factor column
            4. Merge on (time, lat, lon)
            5. Interpolate and save

        Returns: Unified DataFrame with 12+ feature columns
        """
        dfs = []

        # -- Primary: Copernicus --
        if "copernicus" in datasets:
            ds = datasets["copernicus"]
            df_cop = self.copernicus.to_dataframe(ds)

            # Rename to standard lat/lon if needed
            for col in df_cop.columns:
                if col.lower() == "latitude" and "lat" not in df_cop.columns:
                    df_cop = df_cop.rename(columns={col: "lat"})
                if col.lower() == "longitude" and "lon" not in df_cop.columns:
                    df_cop = df_cop.rename(columns={col: "lon"})

            # Coarsen to target_resolution grid
            df_cop["lat_g"] = (df_cop["lat"] / target_resolution).round() * target_resolution
            df_cop["lon_g"] = (df_cop["lon"] / target_resolution).round() * target_resolution
            num_cols = df_cop.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude the grid keys and original lat/lon to avoid duplicate columns after reset_index
            num_cols = [c for c in num_cols if c not in ("lat", "lon", "lat_g", "lon_g")]
            df_cop = (df_cop.groupby(["time", "lat_g", "lon_g"])[num_cols]
                           .mean()
                           .reset_index()
                           .rename(columns={"lat_g": "lat", "lon_g": "lon"}))

            dfs.append(df_cop)
            print(f"\n  [Copernicus Data Processed]")
            print(f"  ↳ Successfully aligned {len(df_cop):,} ocean measurements.")
            print("  ↳ Features Extracted:")
            print("     - Physical: Temperature (thetao), Salinity (so), Ocean Currents (uo, vo), Sea Surface Height (zos)")
            print("     - Biological: Nitrate (no3), Phosphate (po4), Oxygen (o2), Chlorophyll (chl)")
            print("     - Derived Formulas: Current Speed, Nutrient Proxies")
        if "chirps_rain" in datasets:
            ds = datasets["chirps_rain"]
            var_name = next(
                (v for v in ds.data_vars if "prec" in v.lower() or "rain" in v.lower()),
                list(ds.data_vars)[0],
            )
            rain_df = self._ds_to_df(ds, var_name)
            if not rain_df.empty:
                rain_df["seasonal_factor"] = np.clip(rain_df[var_name] / 200.0, 0, 1)
                dfs.append(rain_df[["time", "lat", "lon", "seasonal_factor"]])
                print(f"[Align] CHIRPS Rainfall: {len(rain_df)} rows")

        if not dfs:
            return pd.DataFrame()

        # Merge all on (time, lat, lon)
        merged = dfs[0]
        for df in dfs[1:]:
            merged = merged.merge(df, on=["time", "lat", "lon"], how="left")

        # Interpolate missing values
        merged = merged.sort_values(["lat", "lon", "time"])
        num_cols = merged.select_dtypes(include=[np.number]).columns
        merged[num_cols] = merged[num_cols].interpolate(method="linear")
        merged = merged.dropna(subset=num_cols, how="all")

        # Save
        out = self.output_dir / "unified.parquet"
        merged.to_parquet(out, index=False)
        print(f"\n[OK] Unified dataset: {len(merged)} rows x {len(merged.columns)} columns")
        print(f"     Features: {[c for c in merged.columns if c not in ('time','lat','lon')]}")
        print(f"     Saved to: {out}")

        return merged

    def _ds_to_df(self, ds: xr.Dataset, var_name: str) -> pd.DataFrame:
        """Convert an xarray Dataset to a flat (time, lat, lon, var) DataFrame."""
        try:
            da = ds[var_name]
            coord_map = {}
            for dim in da.dims:
                dl = dim.lower()
                if "lat" in dl or dl == "y":
                    coord_map[dim] = "lat"
                elif "lon" in dl or dl == "x":
                    coord_map[dim] = "lon"
                elif "time" in dl or dl == "t":
                    coord_map[dim] = "time"
            da = da.rename({k: v for k, v in coord_map.items() if k in da.dims})
            df = da.to_dataframe().reset_index()
            keep = [c for c in ["time", "lat", "lon", var_name] if c in df.columns]
            return df[keep].dropna()
        except Exception as e:
            print(f"  [!] Error converting {var_name}: {e}")
            return pd.DataFrame()
