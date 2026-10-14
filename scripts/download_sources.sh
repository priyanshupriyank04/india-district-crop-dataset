#!/usr/bin/env bash
# Step 0: download the two raw inputs that are not fetched by the Python scripts.
#   1. Government of India district crop production statistics (data.gov.in, via a public GitHub mirror)
#   2. India 2020 district boundaries (india_shp_2020)
# SoilGrids and NASA POWER are downloaded by scripts/fetch_soilgrids.py and scripts/fetch_nasa_power.py.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw/crops data/raw/shapefile data/raw/soilgrids data/raw/climate

curl -fsSL -o data/raw/crops/crop_production_datagovin_1997_2015.csv \
  "https://raw.githubusercontent.com/ankitaS11/Crop-Yield-Prediction-in-India-using-ML/master/crop_production.csv"

for ext in shp shx dbf prj cpg; do
  curl -fsSL -o "data/raw/shapefile/in_district.$ext" \
    "https://raw.githubusercontent.com/abhatia08/india_shp_2020/main/district/in_district.$ext"
done
echo "raw sources downloaded into data/raw/"
