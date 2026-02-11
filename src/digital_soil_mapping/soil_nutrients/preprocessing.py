from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform
from sklearn.impute import SimpleImputer


DEFAULT_BAND_NAMES = [
    "elev_m", "slope_deg", "ndvi", "evi", "lst_day_c",
    "aridity_index", "rain_mm", "flowacc_cells", "twi"
]

def to_snake_case(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^\w]+", "_", name)
    name = re.sub(r"__+", "_", name)
    name = name.strip("_")
    return name.lower()

def harmonize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [to_snake_case(c) for c in df.columns]
    return df

def parse_depth_range_to_midpoint(x) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*$", s)
    if m:
        lo = float(m.group(1))
        hi = float(m.group(2))
        return (lo + hi) / 2.0
    try:
        return float(s)
    except Exception:
        return np.nan

def add_profile_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "depth_cm" in df.columns:
        df["depth_cm"] = df["depth_cm"].apply(parse_depth_range_to_midpoint)
        df["depth_cm"] = pd.to_numeric(df["depth_cm"], errors="coerce")
        df["depth_log"] = np.log1p(df["depth_cm"])
    if "horizon_lower" in df.columns:
        df["horizon_lower"] = pd.to_numeric(df["horizon_lower"], errors="coerce")
    if "horizon_upper" in df.columns:
        df["horizon_upper"] = pd.to_numeric(df["horizon_upper"], errors="coerce")
    if "horizon_lower" in df.columns and "horizon_upper" in df.columns:
        df["horizon_thickness"] = df["horizon_lower"] - df["horizon_upper"]
    return df

def sample_multiband_raster_to_df(
    df: pd.DataFrame,
    raster_path: str | Path,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    band_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    df_in = df.copy()
    if lon_col not in df_in.columns or lat_col not in df_in.columns:
        raise ValueError(f"Missing '{lon_col}' and/or '{lat_col}' in dataframe.")
    pts = df_in[[lon_col, lat_col]].astype(float)
    valid_mask = pts[lon_col].notna() & pts[lat_col].notna()
    pts_valid = pts.loc[valid_mask]
    with rasterio.open(str(raster_path)) as src:
        if src.crs is None:
            raise ValueError("Raster has no CRS. Assign CRS before sampling.")
        nbands = src.count
        if band_names is None:
            band_names_use = [f"band_{i}" for i in range(1, nbands + 1)]
        else:
            if len(band_names) != nbands:
                raise ValueError(f"band_names length ({len(band_names)}) != raster band count ({nbands})")
            band_names_use = band_names
        xs = pts_valid[lon_col].to_numpy()
        ys = pts_valid[lat_col].to_numpy()
        raster_crs_str = str(src.crs).upper()
        if raster_crs_str not in ["EPSG:4326", "WGS84", "CRS:84"]:
            xs_t, ys_t = transform("EPSG:4326", src.crs, xs.tolist(), ys.tolist())
            coords = list(zip(xs_t, ys_t))
        else:
            coords = list(zip(xs, ys))
        samples = np.array([v for v in src.sample(coords)], dtype="float64")
        nodata = src.nodata
        if nodata is not None:
            samples[samples == nodata] = np.nan
        cov_df = pd.DataFrame(samples, columns=band_names_use, index=pts_valid.index)
    return df_in.join(cov_df)

def build_feature_map(base_features: List[str]) -> Dict[str, List[str]]:
    fm = {
        # Exchangeable bases / leaching / redistribution
        "ca": base_features + ["ph", "rain_mm", "aridity_index", "twi", "flowacc_cells", "slope_deg", "c_organic", "c_total"],
        "mg": base_features + ["ph", "rain_mm", "aridity_index", "twi", "flowacc_cells", "slope_deg", "c_organic", "c_total"],
        "k":  base_features + ["ph", "c_organic", "c_total", "ndvi", "evi", "rain_mm", "aridity_index", "twi", "flowacc_cells"],
        "na": base_features + ["ph", "electrical_conductivity", "aridity_index", "rain_mm", "twi", "flowacc_cells", "lst_day_c"],

        # Acidity
        "al": base_features + ["ph", "rain_mm", "aridity_index", "twi", "flowacc_cells", "c_organic", "depth_log"],

        # OM-driven
        "n":  base_features + ["c_organic", "c_total", "ndvi", "evi", "lst_day_c", "rain_mm", "aridity_index", "twi", "depth_log"],
        "s":  base_features + ["c_organic", "c_total", "ndvi", "evi", "rain_mm", "aridity_index", "twi", "flowacc_cells", "depth_log"],

        # Redox/drainage sensitive
        "fe": base_features + ["twi", "flowacc_cells", "slope_deg", "rain_mm", "aridity_index", "lst_day_c", "ph", "depth_log"],
        "mn": base_features + ["twi", "flowacc_cells", "slope_deg", "rain_mm", "aridity_index", "lst_day_c", "ph", "depth_log"],

        # Micronutrients
        "zn": base_features + ["ph", "c_organic", "c_total", "twi", "flowacc_cells", "rain_mm", "aridity_index", "ndvi", "evi", "lst_day_c", "depth_log"],
        "cu": base_features + ["c_organic", "c_total", "ph", "twi", "flowacc_cells", "rain_mm", "aridity_index", "ndvi", "evi", "depth_log"],
        "b":  base_features + ["rain_mm", "aridity_index", "twi", "flowacc_cells", "ph", "c_organic", "lst_day_c", "depth_log"],

        # Phosphorus
        "p":  base_features + ["ph", "c_organic", "c_total", "twi", "flowacc_cells", "slope_deg", "rain_mm", "aridity_index", "ndvi", "evi", "depth_log"],
    }

    # Co-nutrient predictors for the micronutrient panel subset
    fm["zn"] += ["b", "s", "p", "na"]
    fm["b"]  += ["zn", "s", "p", "na"]
    fm["p"]  += ["zn", "b", "s"]
    fm["na"] += ["electrical_conductivity"]  # emphasize EC for salinity/sodicity
    for k, cols in fm.items():
        seen = set()
        fm[k] = [c for c in cols if not (c in seen or seen.add(c))]
    return fm

def build_datasets_by_target(
    df: pd.DataFrame,
    feature_map: Dict[str, List[str]],
    drop_predictors: Optional[List[str]] = None,
    impute_strategy: str = "median",
) -> Dict[str, pd.DataFrame]:
    drop_predictors = drop_predictors or []
    datasets: Dict[str, pd.DataFrame] = {}
    for target, features in feature_map.items():
        cols = [target] + features
        cols = [c for c in cols if c in df.columns] 
        df_t = df.loc[df[target].notna(), cols].copy()
        for dp in drop_predictors:
            if dp in df_t.columns and dp != target:
                df_t = df_t.drop(columns=[dp])
        y = df_t[target].reset_index(drop=True)
        X = df_t.drop(columns=[target]).reset_index(drop=True)
        imputer = SimpleImputer(strategy=impute_strategy)
        X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        datasets[target] = pd.concat([X_imp, y], axis=1)
    return datasets


def run_preprocessing(
    train_csv: str | Path,
    raster_tif: str | Path,
    base_features: List[str],
    band_names: Optional[List[str]] = None,
    drop_predictors: Optional[List[str]] = None,
    impute_strategy: str = "median",
) -> Dict[str, pd.DataFrame]:
    band_names = band_names or DEFAULT_BAND_NAMES
    df = pd.read_csv(train_csv)
    df = harmonize_columns(df)
    df = add_profile_features(df)
    df = sample_multiband_raster_to_df(
        df=df,
        raster_path=raster_tif,
        lon_col="longitude",
        lat_col="latitude",
        band_names=band_names,
    )

    feature_map = build_feature_map(base_features)
    dfs_clean = build_datasets_by_target(
        df=df,
        feature_map=feature_map,
        drop_predictors=drop_predictors or ["c_total"],
        impute_strategy=impute_strategy,
    )
    return dfs_clean
