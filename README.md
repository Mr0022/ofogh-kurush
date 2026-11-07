# ofogh-kurush

Daily retail demand panel (`candidate_dataset.parquet`): 1,235 products × 732 days (Jalali 1403/06/31 → 1405/06/31).

## Notebooks

[`notebooks/eda_and_preprocessing.ipynb`](notebooks/eda_and_preprocessing.ipynb) does the following:

1. Converts Jalali dates to Gregorian and profiles the data and the product hierarchy.
2. Audits data quality. It finds 1,499 hidden missing rows, `price = 0` on every zero-sale day, and 6 corrupted dates where `stock_qty` was overwritten by `sale_qty`.
3. Runs EDA: trend, Iranian calendar effects (Nowruz, Ashura, the Thursday peak), demand classes, inflation, discount and availability effects, and autocorrelation.
4. Handles missing values: zero-fill with flags, interpolation, and log-linear price imputation.
5. Handles outliers: replaces price artefacts and caps unexplained sales spikes with a market-adjusted Hampel filter. Raw values are kept.
6. Exports model-ready data for **iTransformer / ModernTCN**: a dense `[T, N]` tensor, `log1p` plus train-only per-channel scaling, a 70/10/20 chronological split, and Jalali/holiday time marks.

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/eda_and_preprocessing.ipynb
```

Running the notebook writes the following to `outputs/` (not committed):

| File | Use |
|---|---|
| `sales_wide_log1p.csv` | Time-Series-Library `custom` dataset (`date` + one column per product) |
| `itransformer_moderntcn_data.npz` | scaled tensor, time marks, masks, covariates, scaler stats, split borders |
| `panel_clean.parquet` | cleaned long panel with all quality flags |
| `product_meta.csv` | hierarchy, demand class, scaler stats per product |

### Forecasting: [`notebooks/forecasting_models.ipynb`](notebooks/forecasting_models.ipynb)

This notebook forecasts sales 30 days ahead for every product from a 112-day history, and compares four methods on the same three rolling 30-day test windows:

1. **Baseline:** same weekday last week.
2. **LightGBM:** global direct multi-horizon model.
3. **ModernTCN:** official model class from `Mr0022/ModernTCN`, cloned automatically.
4. **TiDE.**

All methods are scored by WAPE against actual sales. The notebook is ready for Colab: choose a GPU runtime and run all cells. Set the environment variable `SMOKE_TEST=1` for a one-epoch check.
