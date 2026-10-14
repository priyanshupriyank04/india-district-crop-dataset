"""Merge crop labels + soil + climate into the final district-year-season dataset."""
import pandas as pd, numpy as np
lab = pd.read_csv("data/processed/crop_labels_district_year_season.csv")
soil = pd.read_csv("data/processed/soil_features_district.csv")
clim = pd.read_csv("data/processed/climate_features_district_year_season.csv")
lab['dominant_crop']=lab.dominant_crop.replace({'Paddy':'Rice'})
df = lab.merge(soil.drop(columns=['stname','dtname','Dist_LGD']), on='district_key', how='left') \
        .merge(clim, on=['district_key','Crop_Year','Season'], how='left')
df = df.rename(columns={'shp_stname':'state','shp_dtname':'district','Crop_Year':'crop_year','Season':'season'})
front=['district_key','Dist_LGD','state','district','crop_year','season','centroid_lat','centroid_lon','elevation_m']
feat=[c for c in df.columns if c.startswith('soil_')]+['temp_mean_c','temp_max_c','temp_min_c','humidity_pct','rainfall_mm',
      'rain_days_proxy_mm_per_day','solar_mj_m2_day','wind_m_s','annual_rainfall_mm','annual_temp_mean_c','annual_humidity_pct']
lab_cols=['dominant_crop','dominant_share','dominant_yield','top3_crops','crops_share_ge_10pct','n_crops','total_area_ha']
df=df[front+feat+lab_cols]
df.to_csv("data/final/india_district_crop_dataset_1997_2015.csv", index=False)
# Kaggle-comparable compact version (all seasons, features closest to N/temp/humidity/pH/rainfall)
compact = df[['district_key','state','district','crop_year','season','soil_total_n_g_kg_0_30cm','soil_ph_h2o_0_30cm','soil_soc_g_kg_0_30cm',
              'soil_cec_cmol_kg_0_30cm','soil_clay_pct_0_30cm','soil_sand_pct_0_30cm','temp_mean_c','humidity_pct','rainfall_mm','dominant_crop']]
compact.to_csv("data/final/india_district_crop_dataset_compact.csv", index=False)
print(df.shape); print(df.isna().mean().round(3).to_string()); print(df.dominant_crop.value_counts().head(30))
