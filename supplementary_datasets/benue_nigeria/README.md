# Crop Recommendation Dataset, Yandev District, Benue State, Nigeria

- **Source:** Zenodo, https://doi.org/10.5281/zenodo.19709807 (published 2026-04-23)
- **File here:** `benue_original_dataset.csv`, the record's `original_dataset.csv`, unmodified. `data_dictionary.csv` is the record's own dictionary.
- **Rows / columns:** 5,000 × 9 (`N, P, K, temperature, humidity, ph, rainfall, soil, crop_label`)
- **Classes:** 10 (soybean, beans, guinea corn, pepper, orange, tomatoes, yam, maize, rice, cassava)
- **Licence:** CC BY 4.0. Cite the Zenodo record when using it.

## Provenance (from the record)

2,200 samples originally collected by Kwaghtyo et al., plus 2,800 collected by random sampling across farm fields in Yandev District, Gboko LGA, in the 2024 and 2025 seasons. Soil sampled at 0–30 cm and analysed in a laboratory. The same record also publishes SMOTE, VAE and GAN-generated variants; only the original file is included here.

## Note

In our nested cross-validation every classifier reached macro-F1 ≥ 0.998, including logistic regression, so the classes are close to linearly separable on these seven features.
