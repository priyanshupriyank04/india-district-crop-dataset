# India District Crop Dataset (1997-2015) — Data Card

Built 2026-09-25 from free, open, no-registration sources. Every row is one
district × agricultural year × season, with soil and climate features and the
crop actually grown (observed label), so it can replace or complement the
2,200-row Kaggle "Crop Recommendation" dataset of unknown provenance.

## Files

| File | Rows | Cols | Description |
|---|---|---|---|
| `data/final/india_district_crop_dataset_1997_2015.csv` | 34,903 | 48 | Full dataset (24 MB) |
| `data/final/india_district_crop_dataset_compact.csv` | 34,903 | 15 | Kaggle-like subset (N, pH, SOC, CEC, texture, temp, humidity, rainfall → crop) |
| `data/processed/crop_long_1997_2015.csv` | 246,091 | 20 | One row per district-year-season-crop (area, production, yield, area share) for multi-label or yield tasks |
| `data/processed/district_crosswalk.csv` | 652 | 8 | Crop-table district name → 2020 polygon (Census 2011 code, LGD code), with match method |

## Sources (all free, no login)

| Component | Source | Access | Licence | Notes |
|---|---|---|---|---|
| Crop area/production (label) | Government of India, DES/data.gov.in "District-wise, season-wise crop production statistics", 1997-98 to 2014-15 | GitHub mirror of the data.gov.in CSV (`ankitaS11/Crop-Yield-Prediction-in-India-using-ML`) | Government Open Data License India (GODL) | 246,091 rows, 33 states, 646 districts, 124 crops |
| District boundaries | `abhatia08/india_shp_2020` (735 districts, Census 2011 + LGD codes) | GitHub | open | Used for zonal statistics and centroids |
| Soil | ISRIC SoilGrids 2.0 (pH, total N, SOC, CEC, clay, sand, silt, bulk density, OC density, coarse fragments) at 0-5, 5-15, 15-30 cm | WCS `maps.isric.org`, clipped to India at 1 km | CC BY 4.0 | Zonal mean per district polygon; 0-30 cm is depth-weighted mean |
| Climate | NASA POWER monthly (T2M, T2M_MAX, T2M_MIN, RH2M, PRECTOTCORR, ALLSKY_SFC_SW_DWN, WS2M), 1997-2016, at district representative point | `power.larc.nasa.gov` API | public domain | ~0.5° × 0.625° resolution; elevation from POWER geometry |

## Label definition

`dominant_crop` = crop with largest sown area in that district-year-season,
after excluding plantation crops (coconut, arecanut, tea, coffee, rubber,
cashew) and aggregate categories ("Other Kharif pulses", "Total foodgrain"…).
`Paddy` merged into `Rice`. `dominant_share` is its share of season area;
`top3_crops` and `crops_share_ge_10pct` support multi-label formulations;
`dominant_yield` (t/ha) is an auxiliary regression target.

`Crop_Year` Y means the agricultural year Y/Y+1. Season climate windows:
Kharif Jun-Oct Y; Rabi Nov Y - Mar Y+1; Whole Year Jun Y - May Y+1;
Summer Mar-May Y+1; Autumn Sep-Nov Y; Winter Dec Y - Feb Y+1.
`rainfall_mm` is the season total; `annual_*` columns cover Jun Y - May Y+1.

## Feature list

Soil (per district, static): `soil_ph_h2o`, `soil_total_n_g_kg`, `soil_soc_g_kg`,
`soil_cec_cmol_kg`, `soil_clay_pct`, `soil_sand_pct`, `soil_silt_pct`,
`soil_bulk_density_kg_dm3`, `soil_org_c_density_kg_m3`, `soil_coarse_frag_pct`,
each at `_0_5cm` and `_0_30cm`.

Climate (per district-year-season): `temp_mean_c`, `temp_max_c`, `temp_min_c`,
`humidity_pct`, `rainfall_mm`, `rain_days_proxy_mm_per_day`, `solar_mj_m2_day`,
`wind_m_s`, `annual_rainfall_mm`, `annual_temp_mean_c`, `annual_humidity_pct`.

Location: `centroid_lat`, `centroid_lon`, `elevation_m`, `state`, `district`,
`Dist_LGD`, `district_key` (Census 2011 state_district code).

## Missing values

Feature columns (soil, climate, location) have **zero** missing values. The only
gaps are: `dominant_yield` is blank in 45 rows (production not reported in the
source table), `crops_share_ge_10pct` is blank in 1 row, and
`soil_filled_from_nearest` is a flag column that is empty except for Mumbai
(urban polygon with no soil pixels; soil copied from Mumbai Suburban).

## Known limitations (state these in the paper)

1. **No P or K.** SoilGrids has no phosphorus, potassium, EC or micronutrients.
   Total N (SoilGrids) is not the same quantity as Soil Health Card "available N".
   The Kaggle N/P/K columns are therefore not directly comparable.
2. **District aggregation.** Soil is a district-mean of a 1 km grid; climate is
   at one representative point per district. Within-district variation is lost.
3. **Boundary vintage.** Crop statistics use 1997-2015 district names; polygons
   are 2020 vintage. Districts split after 2011 (e.g. Warangal, Bardhaman) are
   mapped to one successor polygon; see `match_method` in the crosswalk.
4. **Observed labels encode economics.** The dominant crop reflects irrigation,
   markets and MSP as much as agronomy; discuss, and consider GAEZ agreement as
   an external check (GAEZ is CC BY-NC-SA, so use only for validation).
5. **Spatial/temporal autocorrelation.** Rows from the same district across
   years are near-duplicates in soil features. Use GroupKFold by district
   (or state / agro-climatic zone) and a year-based split; never random k-fold
   alone.
6. **Class imbalance.** Rice dominates (~30 %). Report macro-F1 and per-class.
7. **NASA POWER humidity** is coarse (~55 km); IMD does not provide free
   gridded humidity.

## Reproduction

```
python -m venv .venv && .venv/bin/pip install pandas numpy requests geopandas rasterio rasterstats scikit-learn
.venv/bin/python scripts/fetch_soilgrids.py          # ~30 GeoTIFFs, ~40 min
.venv/bin/python scripts/fetch_nasa_power.py         # 735 API calls, ~60 min
.venv/bin/python scripts/build_district_crosswalk.py
.venv/bin/python scripts/build_crop_labels.py
.venv/bin/python scripts/build_soil_features.py
.venv/bin/python scripts/build_climate_features.py
.venv/bin/python scripts/build_merged_dataset.py
```

## Sanity baseline (Random Forest, 300 trees, 32 classes with >=100 rows, 34,270 rows)

| Cross-validation | Accuracy |
|---|---|
| Random stratified 5-fold | 0.824 |
| GroupKFold by district | 0.709 |
| GroupKFold by state | 0.526 |
| Majority class (Rice) | 0.309 |

The gap between random and grouped folds is the leakage effect reviewers ask about.
