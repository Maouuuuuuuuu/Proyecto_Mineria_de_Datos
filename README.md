# Sales and Demand Forecasting

This project uses OLIST sales data to forecast weekly sales revenue for the company and its regions, and product demand by region and category.

The Python workflow compares forecasts with actual sales, calculates error metrics, and generates a standalone HTML report with three views:

- **Global:** company-wide sales value in BRL.
- **Regions:** sales value for each region.
- **Region + Category:** expected units sold for each product category within a region.

The final report includes **Naive, Random Forest (RF), and Prophet**. Sales revenue is based on item prices; it is not profit.

## How the models were selected

We tested **Naive, RF, ARIMAX, SARIMAX, and Prophet** using expanding-window walk-forward validation. Each model began with 52 training weeks, predicted the next two weeks, then repeated the process with the newly available history. The evaluation covered 25 rounds and 50 forecast weeks, without random train/test splitting.

Models were compared using **MAE**, **RMSE**, and **WAPE**. Lower values are better. MAE measures the average absolute weekly error; RMSE gives more weight to large errors; WAPE expresses accumulated absolute error relative to actual sales.

### Global sales

Results across the 50 evaluated weeks:

| Model | MAE (R$) | RMSE (R$) | WAPE (%) |
| --- | ---: | ---: | ---: |
| Naive | 40,431.00 | 59,223.37 | 20.27 |
| RF | 39,582.67 | 59,727.63 | 19.85 |
| ARIMAX | 40,066.47 | 60,035.66 | 20.09 |
| SARIMAX | 39,083.76 | **57,237.13** | 19.60 |
| Prophet | **37,873.69** | 58,944.39 | **18.99** |

![Global sales demo: actual versus predicted sales and model errors](images/Global%20sales%20demo%20result.png)

### Regional sales

Arithmetic averages of the five regions' scores, giving each region equal weight. These are regional averages, not company-wide pooled errors; the WAPE column is the mean of regional WAPEs.

| Model | Mean MAE (R$) | Mean RMSE (R$) | Mean WAPE (%) |
| --- | ---: | ---: | ---: |
| Naive | 9,399.84 | 13,290.56 | 29.47 |
| RF | 9,099.86 | 13,092.70 | 25.23 |
| ARIMAX | 8,982.15 | 13,074.00 | 26.08 |
| SARIMAX | 8,618.56 | 12,703.91 | 26.20 |
| Prophet | 8,551.20 | 12,681.88 | 25.30 |

The regional example below shows **Sul**, rather than the five-region averages above.

![Sul regional sales demo: actual versus predicted sales and model errors](images/Region%20sales%20demo%20result.png)

### Demand by region and category

Example: **Sudeste / papelaria** (stationery), measured in units. This illustrates one combination, not an average across all categories.

![Sudeste stationery demand demo: actual versus predicted units and model errors](images/Product%20category%20demo%20result.png)

**Prophet** achieved the lowest global MAE and the lowest MAE in Sudeste and Sul. **RF** achieved the lowest MAE in Centro-Oeste, Nordeste, and Norte. We retained these two models to compare their behavior across different demand patterns, with **Naive** as the reference baseline. SARIMAX remained competitive and had the lowest global RMSE; the selection does not imply that RF and Prophet win every metric or category.

The evaluation notebooks contain the model configurations and comparisons:

- [Global sales](Demos/Global%20walkforward.ipynb)
- [Regional sales](Demos/Regional%20walkforward.ipynb)
- [Units by region and category](Demos/Region%20category%20walkforward.ipynb) — evaluated 347 combinations; results are inspected by combination.

To run a demo, open its notebook in Jupyter or VS Code, select your Python environment, configure the SQL connection, and run the cells in order. Saved tables and plots can be viewed on GitHub without rerunning training. The demos additionally require `matplotlib`, `statsmodels`, and a Jupyter-compatible environment.

## Running the project

The repository includes the evaluation notebooks and their saved demo results. The final `forecast_report.py` script will be added separately; the report-generation instructions below describe the local workflow.

Use a Python environment with the required packages and install **Microsoft ODBC Driver 18 for SQL Server** on the computer running the script.

```powershell
python -m pip install pyodbc pandas numpy scikit-learn prophet
```

The script expects `gold.dim_demand`, with **one row per sold item** and these columns:

```text
region, category, purchase_year, purchase_week, price
```

The table must already contain the intended sales records; the query does not filter order status. Missing categories are retained as `unknown_category`. Edit `CONNECTION` in `forecast_report.py` to point to your SQL Server and database. The supplied configuration uses Windows authentication.

From the folder containing the script, run:

```powershell
python forecast_report.py
```

The script evaluates the models, refits on all eligible history, and forecasts the next four weeks. It saves results locally and does not write them to SQL.

## Opening the report

The script creates these outputs beside itself:

```text
forecast_results/
    results.json
    checkpoints/
reports/
    rf_prophet_report.html
```

**Open `reports/rf_prophet_report.html` directly in your browser.** It contains the charts, data, and filters, so viewing or sharing it requires no Python, SQL connection, internet connection, or retraining.

A normal rerun reuses matching checkpoints. To rebuild only the HTML from saved results:

```powershell
python forecast_report.py --report-only
```

Use `--refresh` to force retraining, or `--workers 1` to reduce simultaneous processing. Training and forecast horizons can be changed near the top of the script.

## Interpreting the results

Metrics describe historical evaluation; the next-four-week forecasts have no measured errors yet. **Total test sales/units** is the cumulative actual amount across the evaluation period. Future dates follow the dataset cutoff, not today's date.

Missing weeks are treated as zero, the final recorded week is excluded by default, and dates use the source year/week labels as an ISO-week approximation. WAPE is undefined when actual demand totals zero. Constant training histories use a labeled constant forecast, and failed predictions are not treated as zero.

These results support comparing the tested configurations on this dataset; they do not guarantee future accuracy.

