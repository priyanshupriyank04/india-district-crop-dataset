# Kaggle Crop Recommendation Dataset (Ingle, 2020)

- **Source:** https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
- **Rows / columns:** 2,200 × 8 (`N, P, K, temperature, humidity, ph, rainfall, label`)
- **Classes:** 22 crops, exactly 100 rows each
- **Licence:** none stated on the source page, so the file is **not redistributed** here.

## Get the file

```bash
bash supplementary_datasets/kaggle_ingle/download.sh
```

This fetches a public mirror into `supplementary_datasets/kaggle_ingle/Crop_recommendation.csv`. You can also download it from Kaggle directly with a Kaggle account.

## Provenance note

The Kaggle page says the data "was build by augmenting datasets of rainfall, climate and fertilizer data available for India". No sampling protocol or generation code is published, the N/P/K units are unspecified, and later papers attribute it to incompatible origins. It is used in this project only as the benchmark of the earlier, rejected study.
