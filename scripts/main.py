import data_preparation
import pandas as pd
from pathlib import Path

def main() -> None:
    train_path = Path("../data/train/Train.csv")
    raster_path = Path("../data/raw/Africa_Geospatial_Covariates_2008_2018_5km.tif")
    out_json = Path("../data/processed/train_datasets_by_target.json")
    band_names = [
        "elev_m", "slope_deg", "ndvi", "evi", "lst_day_c",
        "aridity_index", "rain_mm", "flowacc_cells", "twi"
    ]
    df = pd.read_csv(train_path)
    df = data_preparation.harmonize_columns(df)
    df = data_preparation.add_profile_features(df)
    df = data_preparation.sample_multiband_raster_to_df(
        df=df,
        raster_path=raster_path,
        lon_col="longitude",
        lat_col="latitude",
        band_names=band_names
    )
    base_features = [
        'latitude',
        'longitude',
        "depth_cm",
        "depth_log",
        "horizon_upper",
        "horizon_lower",
        "horizon_thickness",
        "ph",
        "electrical_conductivity",
        "c_organic",
        "c_total",
        "elev_m",
        "slope_deg",
        "ndvi",
        "evi",
        "lst_day_c",
        "aridity_index",
        "rain_mm",
        "flowacc_cells",
        "twi",
    ]
    feature_map = data_preparation.build_feature_map(base_features)
    datasets = data_preparation.build_datasets_by_target(
        df=df,
        feature_map=feature_map,
        drop_predictors=["c_total"],
        impute_strategy="median"
    )
    data_preparation.save_one_json(out_json, feature_map, datasets)
    print(f"Saved: {out_json.resolve()}")
    counts = {t: len(d) for t, d in datasets.items()}
    print("Rows per target:", counts)

if __name__ == "__main__":
    main()
