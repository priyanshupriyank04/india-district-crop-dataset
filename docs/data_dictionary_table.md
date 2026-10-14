| Column | Type | Unit | Source | Description | Model input |
|---|---|---|---|---|---|
| `district_key` | text | - | india_shp_2020 | Census-2011 state code + district code, e.g. 09_146. Primary district identifier. | group key |
| `Dist_LGD` | integer | - | india_shp_2020 | Local Government Directory (LGD) district code | no |
| `state` | text | - | india_shp_2020 | State or union territory (2020 boundaries) | no |
| `district` | text | - | india_shp_2020 | District name (2020 boundaries) | no |
| `crop_year` | integer | year | data.gov.in DES | Agricultural year Y means Y/Y+1 (1997-2015) | no |
| `season` | text | - | data.gov.in DES | Kharif, Rabi, Whole Year, Summer, Autumn or Winter | yes (one-hot) |
| `centroid_lat` | float | degrees N | india_shp_2020 | Latitude of district representative point | no (would memorise location) |
| `centroid_lon` | float | degrees E | india_shp_2020 | Longitude of district representative point | no (would memorise location) |
| `elevation_m` | float | m | NASA POWER | Elevation of the district representative point | yes |
| `soil_ph_h2o_0_5cm` | float | pH | ISRIC SoilGrids 2.0 | Soil pH in water, 0-5 cm, district mean of 1 km pixels | no (near-duplicate of 0-30 cm) |
| `soil_ph_h2o_0_30cm` | float | pH | ISRIC SoilGrids 2.0 | Soil pH in water, 0-30 cm (depth-weighted), district mean of 1 km pixels | yes |
| `soil_total_n_g_kg_0_5cm` | float | g/kg | ISRIC SoilGrids 2.0 | Total nitrogen, 0-5 cm, district mean of 1 km pixels | no (near-duplicate of 0-30 cm) |
| `soil_total_n_g_kg_0_30cm` | float | g/kg | ISRIC SoilGrids 2.0 | Total nitrogen, 0-30 cm (depth-weighted), district mean of 1 km pixels | yes |
| `soil_soc_g_kg_0_5cm` | float | g/kg | ISRIC SoilGrids 2.0 | Soil organic carbon content, 0-5 cm, district mean of 1 km pixels | no (near-duplicate of 0-30 cm) |
| `soil_soc_g_kg_0_30cm` | float | g/kg | ISRIC SoilGrids 2.0 | Soil organic carbon content, 0-30 cm (depth-weighted), district mean of 1 km pixels | yes |
| `soil_cec_cmol_kg_0_5cm` | float | cmol(c)/kg | ISRIC SoilGrids 2.0 | Cation exchange capacity, 0-5 cm, district mean of 1 km pixels | no (near-duplicate of 0-30 cm) |
| `soil_cec_cmol_kg_0_30cm` | float | cmol(c)/kg | ISRIC SoilGrids 2.0 | Cation exchange capacity, 0-30 cm (depth-weighted), district mean of 1 km pixels | yes |
| `soil_clay_pct_0_5cm` | float | % | ISRIC SoilGrids 2.0 | Clay content, 0-5 cm, district mean of 1 km pixels | no (near-duplicate of 0-30 cm) |
| `soil_clay_pct_0_30cm` | float | % | ISRIC SoilGrids 2.0 | Clay content, 0-30 cm (depth-weighted), district mean of 1 km pixels | yes |
| `soil_sand_pct_0_5cm` | float | % | ISRIC SoilGrids 2.0 | Sand content, 0-5 cm, district mean of 1 km pixels | no (near-duplicate of 0-30 cm) |
| `soil_sand_pct_0_30cm` | float | % | ISRIC SoilGrids 2.0 | Sand content, 0-30 cm (depth-weighted), district mean of 1 km pixels | yes |
| `soil_silt_pct_0_5cm` | float | % | ISRIC SoilGrids 2.0 | Silt content, 0-5 cm, district mean of 1 km pixels | no (near-duplicate of 0-30 cm) |
| `soil_silt_pct_0_30cm` | float | % | ISRIC SoilGrids 2.0 | Silt content, 0-30 cm (depth-weighted), district mean of 1 km pixels | yes |
| `soil_bulk_density_kg_dm3_0_5cm` | float | kg/dm3 | ISRIC SoilGrids 2.0 | Bulk density of fine earth, 0-5 cm, district mean of 1 km pixels | no (near-duplicate of 0-30 cm) |
| `soil_bulk_density_kg_dm3_0_30cm` | float | kg/dm3 | ISRIC SoilGrids 2.0 | Bulk density of fine earth, 0-30 cm (depth-weighted), district mean of 1 km pixels | yes |
| `soil_org_c_density_kg_m3_0_5cm` | float | kg/m3 | ISRIC SoilGrids 2.0 | Organic carbon density, 0-5 cm, district mean of 1 km pixels | no (near-duplicate of 0-30 cm) |
| `soil_org_c_density_kg_m3_0_30cm` | float | kg/m3 | ISRIC SoilGrids 2.0 | Organic carbon density, 0-30 cm (depth-weighted), district mean of 1 km pixels | yes |
| `soil_coarse_frag_pct_0_5cm` | float | % | ISRIC SoilGrids 2.0 | Coarse fragments (volumetric), 0-5 cm, district mean of 1 km pixels | no (near-duplicate of 0-30 cm) |
| `soil_coarse_frag_pct_0_30cm` | float | % | ISRIC SoilGrids 2.0 | Coarse fragments (volumetric), 0-30 cm (depth-weighted), district mean of 1 km pixels | yes |
| `soil_filled_from_nearest` | text | - | derived | Provenance flag: non-empty only for Mumbai, whose soil was copied from the named neighbour | no |
| `temp_mean_c` | float | degC | NASA POWER | Mean 2 m air temperature over the season window | yes |
| `temp_max_c` | float | degC | NASA POWER | Mean of monthly maximum 2 m temperature over the season window | yes |
| `temp_min_c` | float | degC | NASA POWER | Mean of monthly minimum 2 m temperature over the season window | yes |
| `humidity_pct` | float | % | NASA POWER | Mean 2 m relative humidity over the season window | yes |
| `rainfall_mm` | float | mm | NASA POWER | Total precipitation over the season window | yes |
| `rain_days_proxy_mm_per_day` | float | mm/day | NASA POWER | Mean daily precipitation over the season window | yes |
| `solar_mj_m2_day` | float | MJ/m2/day | NASA POWER | Mean all-sky surface shortwave radiation over the season window | yes |
| `wind_m_s` | float | m/s | NASA POWER | Mean 2 m wind speed over the season window | yes |
| `annual_rainfall_mm` | float | mm | NASA POWER | Total precipitation, June of crop_year to May of the next year | yes |
| `annual_temp_mean_c` | float | degC | NASA POWER | Mean temperature, June of crop_year to May of the next year | yes |
| `annual_humidity_pct` | float | % | NASA POWER | Mean relative humidity, June of crop_year to May of the next year | yes |
| `dominant_crop` | text | - | data.gov.in DES | TARGET. Crop with the largest sown area in this district-year-season | target |
| `dominant_share` | float | fraction | derived | Dominant crop's share of the season's sown area | no (label leakage) |
| `dominant_yield` | float | t/ha* | derived | Production / area of the dominant crop. *Source mixes units; 45 blanks | no (label leakage) |
| `top3_crops` | text | - | derived | Three largest crops by area, separated by | | no (label leakage) |
| `crops_share_ge_10pct` | text | - | derived | All crops with >=10% of season area, separated by | (multi-label target) | no (label leakage) |
| `n_crops` | integer | count | derived | Number of crops reported that season | no (label leakage) |
| `total_area_ha` | float | ha | data.gov.in DES | Total sown area of the included crops that season | no (label leakage) |