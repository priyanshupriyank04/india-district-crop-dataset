# Supplementary datasets

These three public crop-recommendation datasets are kept alongside the main India dataset for comparison and future external-validation work. They are **not** part of the India District Crop Dataset, were not built by this project, and keep their original licences. Files are unmodified copies of the originals.

| Folder | Region | Rows | Features | Classes | Licence | Included here |
|---|---|---|---|---|---|---|
| [`kaggle_ingle/`](kaggle_ingle/) | Unspecified ("India", augmented) | 2,200 | 7 (N, P, K, temperature, humidity, pH, rainfall) | 22 | Not stated | Download script only |
| [`benue_nigeria/`](benue_nigeria/) | Yandev, Benue State, Nigeria | 5,000 | 7 + soil texture | 10 | CC BY 4.0 | Yes |
| [`ethiopia/`](ethiopia/) | Ethiopia | 3,867 | 27 (soil chemistry + seasonal NASA POWER climate) + soil colour | 12 | CC BY-NC-ND 4.0 | Yes, unmodified |

## Caveats observed in our experiments

- **Kaggle (Ingle):** exactly 100 rows per class, described on its own page as "built by augmenting" other data; every classifier scores 0.97–0.99. Treat it as a saturated benchmark, not field data.
- **Benue, Nigeria:** declared field-collected, but every model, including logistic regression, separates it almost perfectly (macro-F1 ≥ 0.998).
- **Ethiopia:** genuinely hard; nested-CV macro-F1 0.16–0.24 across five classifiers.

None of the three has location keys that join with the India dataset.
