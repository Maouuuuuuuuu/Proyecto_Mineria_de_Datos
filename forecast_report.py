from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import hashlib
import inspect
import json
import logging
import math
import os
import warnings


INITIAL_TRAIN_WEEKS = 52
HORIZON_WEEKS = 2
FUTURE_WEEKS = 4
EXCLUDE_LAST_WEEK = True
WORKERS = 4
INTERVAL_WIDTH = 0.80
UNCERTAINTY_SAMPLES = 1000
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / 'forecast_results'
REPORT_FILE = BASE_DIR / 'reports' / 'rf_prophet_report.html'
CONNECTION = (
    'DRIVER={ODBC Driver 18 for SQL Server};'
    r'SERVER=localhost\MSSQLSERVER01;'
    'DATABASE=Proyecto_Final_MD;'
    'Trusted_Connection=yes;Encrypt=yes;TrustServerCertificate=yes;'
)


def initialize_models():
    global np, pd, RandomForestRegressor, Prophet
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[name] = '1'
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from prophet import Prophet
    logging.getLogger('cmdstanpy').disabled = True


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(text, encoding='utf-8')
    temp.replace(path)


def save_json(path, data):
    atomic_text(path, json.dumps(data, ensure_ascii=False, allow_nan=False))


def read_groups():
    import pyodbc
    conn = pyodbc.connect(CONNECTION, timeout=10)
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT region,
                   CASE WHEN category IS NULL OR UPPER(LTRIM(RTRIM(category))) IN ('', 'NULL', 'NAN')
                        THEN 'unknown_category' ELSE LTRIM(RTRIM(category)) END AS category,
                   purchase_year, purchase_week,
                   SUM(CAST(price AS decimal(19,2))) AS sales, COUNT_BIG(*) AS units
            FROM gold.dim_demand
            GROUP BY region,
                     CASE WHEN category IS NULL OR UPPER(LTRIM(RTRIM(category))) IN ('', 'NULL', 'NAN')
                          THEN 'unknown_category' ELSE LTRIM(RTRIM(category)) END,
                     purchase_year, purchase_week
        """)
        df = pd.DataFrame.from_records([tuple(r) for r in cur.fetchall()],
                                      columns=[c[0] for c in cur.description])
    finally:
        conn.close()
    if df.empty or df.region.isna().any() or df.region.astype(str).str.strip().eq('').any():
        raise ValueError('Empty source or missing region labels.')
    for column in ['sales', 'units']:
        df[column] = df[column].astype(float)
        if not np.isfinite(df[column]).all() or df[column].lt(0).any():
            raise ValueError(f'Invalid {column} values.')

    df['week_start'] = pd.to_datetime(
        df.purchase_year.astype(int).astype(str) + '-W'
        + df.purchase_week.astype(int).astype(str).str.zfill(2) + '-1', format='%G-W%V-%u')
    if df.duplicated(['region', 'category', 'week_start']).any():
        raise ValueError('Duplicate reconstructed region/category/week labels.')

    df['sales_cents'] = (df.sales * 100).round().astype('int64')
    source_last = df.week_start.max().date().isoformat()
    calendar = pd.date_range(df.week_start.min(), df.week_start.max(), freq='W-MON')
    if EXCLUDE_LAST_WEEK:
        calendar = calendar[:-1]
    if not 12 <= INITIAL_TRAIN_WEEKS < len(calendar):
        raise ValueError('Need at least 12 initial training weeks and some evaluation weeks.')
    if HORIZON_WEEKS < 1 or FUTURE_WEEKS < 1:
        raise ValueError('Forecast horizons must be positive.')
    groups = []
    def add(scope, region, category, series):
        aligned = series.reindex(calendar, fill_value=0)
        groups.append({'scope': scope, 'region': region, 'category': category,
                       'dates': [d.date().isoformat() for d in calendar],
                       'values': aligned.tolist(), 'source_last_week': source_last})
    add('global', 'GLOBAL', '', df.groupby('week_start').sales_cents.sum() / 100)
    for region, part in df.groupby('region', sort=True):
        add('region', region, '', part.groupby('week_start').sales_cents.sum() / 100)
    for (region, category), part in df.groupby(['region', 'category'], sort=True):
        add('category', region, category, part.groupby('week_start').units.sum())
    return groups


def forecast_rf(train, future_dates):
    table = train.to_frame(name='sales')
    table['month'] = table.index.month
    for lag in range(1, 5):
        table[f'lag_{lag}'] = table.sales.shift(lag)
    table['mean_4'] = table.sales.shift(1).rolling(4).mean()
    table['mean_8'] = table.sales.shift(1).rolling(8).mean()
    table = table.dropna()
    features = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'mean_4', 'mean_8', 'month']
    model = RandomForestRegressor(
        n_estimators=300, max_depth=4, min_samples_leaf=3,
        random_state=42, n_jobs=1,
    )
    model.fit(table[features], table.sales)
    history = train.tolist()
    predictions = []
    for week in future_dates:
        inputs = pd.DataFrame([{
            'lag_1': history[-1], 'lag_2': history[-2],
            'lag_3': history[-3], 'lag_4': history[-4],
            'mean_4': np.mean(history[-4:]), 'mean_8': np.mean(history[-8:]),
            'month': week.month,
        }])
        prediction = float(model.predict(inputs[features])[0])
        predictions.append(prediction)
        history.append(prediction)
    return np.array(predictions)

def month_inputs(dates):
    return pd.DataFrame({
        'month_sin': np.sin(2 * np.pi * (dates.month - 1) / 12),
        'month_cos': np.cos(2 * np.pi * (dates.month - 1) / 12),
    }, index=dates)




def forecast_prophet(train, future_dates):
    scale = max(train.mean(), 1.0)
    training = month_inputs(train.index).reset_index(names='ds')
    training['y'] = train.to_numpy() / scale
    future = month_inputs(future_dates).reset_index(names='ds')
    model = Prophet(yearly_seasonality=False, weekly_seasonality=False,
                    daily_seasonality=False, changepoint_prior_scale=0.05,
                    uncertainty_samples=UNCERTAINTY_SAMPLES, interval_width=INTERVAL_WIDTH)
    for column in ['month_sin', 'month_cos']:
        model.add_regressor(column, standardize=False, prior_scale=1.0)
    model.fit(training, seed=42)
    previous_state = np.random.get_state()
    try:
        np.random.seed(42)
        forecast = model.predict(future)
    finally:
        np.random.set_state(previous_state)
    return forecast[['yhat', 'yhat_lower', 'yhat_upper']].to_numpy() * scale


def predict_checked(model, train, dates):
    """Constant history is explicit; failures stay missing, never replaced by zero."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        bounds = {'lower': [None] * len(dates), 'upper': [None] * len(dates)}
        try:
            constant = train.nunique() == 1
            values = (np.repeat(float(train.iloc[-1]), len(dates)) if constant else
                      (forecast_rf if model == 'RF' else forecast_prophet)(train, dates))
            values = np.asarray(values, dtype=float)
            if model == 'Prophet' and not constant:
                if values.shape != (len(dates), 3):
                    raise ValueError('Invalid interval shape')
                lower, upper = values[:, 1], values[:, 2]
                if not np.isfinite(values).all() or (lower > upper).any():
                    raise ValueError('Invalid interval bounds')
                bounds = {'lower': np.maximum(lower, 0).tolist(),
                          'upper': np.maximum(upper, 0).tolist()}
                values = values[:, 0]
            if values.shape != (len(dates),) or not np.isfinite(values).all():
                raise ValueError('Invalid predictions')
            values = np.maximum(values, 0).tolist()
            status = 'ok'
            message = 'Constant training history; constant forecast, no model fitted.' if constant else ''
        except Exception as exc:
            values, status, message, constant = [None] * len(dates), 'failed', str(exc), False
            bounds = {'lower': [None] * len(dates), 'upper': [None] * len(dates)}
        messages = list(dict.fromkeys(str(w.message) for w in caught))
    return values, status, ' | '.join([message] + messages).strip(' |'), constant, bounds


def evaluate_group(job):
    group, settings = job
    initial, horizon, future_weeks = (settings[k] for k in ['initial', 'horizon', 'future'])
    dates = pd.DatetimeIndex(group['dates'], freq='W-MON')
    sales = pd.Series(group['values'], index=dates, name='sales', dtype=float)
    target = 'units' if group['scope'] == 'category' else 'sales'
    unit = 'units' if target == 'units' else 'brl'
    base = {'region': group['region'], 'category': group['category'],
            'exclude_last_week': settings['exclude_last_week']}
    summary = {**base, 'train_start': group['dates'][0], 'train_end': group['dates'][initial-1],
               'train_weeks': initial, 'test_weeks': len(sales)-initial,
               'test_start': group['dates'][initial], 'test_end': group['dates'][-1],
               'horizon_weeks': horizon, f'actual_{target}_total': float(sales.iloc[initial:].sum())}
    result = {'metrics': [], 'predictions': [], 'future': [], 'diagnostics': []}
    values_by_model = {m: [] for m in ['RF', 'Prophet']}
    constant_rounds = {m: 0 for m in values_by_model}
    for fold, stop in enumerate(range(initial, len(sales), horizon), 1):
        train, test = sales.iloc[:stop], sales.iloc[stop:stop+horizon]
        for model in values_by_model:
            values, status, message, constant, bounds = predict_checked(model, train, test.index)
            constant_rounds[model] += int(constant)
            values_by_model[model].extend(values)
            if message:
                result['diagnostics'].append({**base, 'model': model, 'fold': fold,
                                              'phase': 'evaluation', 'status': status, 'message': message})
            for step, (week, actual, value) in enumerate(zip(test.index, test, values), 1):
                result['predictions'].append({**base, 'model': model,
                    # Initial cutoff groups the complete walk-forward evaluation in the report.
                    'train_end': summary['train_end'], 'forecast_origin': train.index[-1].date().isoformat(),
                    'test_weeks': summary['test_weeks'], 'fold': fold, 'horizon_week': step,
                    'week_start': week.date().isoformat(), f'actual_{target}': float(actual),
                    f'predicted_{target}': value, 'status': status,
                    'lower': bounds['lower'][step-1], 'upper': bounds['upper'][step-1]})
    matrix = np.array(list(values_by_model.values()), dtype=float)
    common = np.isfinite(matrix).all(axis=0)
    actual = sales.iloc[initial:].to_numpy()[common]
    total = float(np.abs(actual).sum())
    for i, model in enumerate(values_by_model):
        error = actual - matrix[i, common]
        result['metrics'].append({**summary, 'model': model, 'scored_weeks': int(common.sum()),
            'successful_weeks': int(np.isfinite(matrix[i]).sum()), 'constant_rounds': constant_rounds[model],
            f'mae_{unit}': float(np.abs(error).mean()) if len(error) else None,
            f'rmse_{unit}': float(np.sqrt(np.mean(error**2))) if len(error) else None,
            'wape_pct': float(100*np.abs(error).sum()/total) if total else None,
            'status': 'ok' if len(error) else 'failed', 'message': ''})
    future_dates = pd.date_range(sales.index[-1]+pd.Timedelta(weeks=1), periods=future_weeks, freq='W-MON')
    for model in values_by_model:
        values, status, message, constant, bounds = predict_checked(model, sales, future_dates)
        if message:
            result['diagnostics'].append({**base, 'model': model, 'phase': 'future', 'status': status, 'message': message})
        for step, (week, value) in enumerate(zip(future_dates, values), 1):
            result['future'].append({**base, 'model': model, 'forecast_type': 'future',
                'train_start': group['dates'][0], 'train_end': group['dates'][-1], 'train_weeks': len(sales),
                'source_last_week': group['source_last_week'], 'forecast_weeks': future_weeks,
                'horizon_week': step, 'week_start': week.date().isoformat(), f'predicted_{target}': value,
                'status': status, 'message': message,
                'lower': bounds['lower'][step-1], 'upper': bounds['upper'][step-1]})
    return result


def add_naive_baseline(group, result):
    """Add/rebuild Naive cheaply, without invalidating trained-model checkpoints.

    At each origin, repeat the last observed value for the entire forecast block.
    Score it on exactly the same successful weeks as RF and Prophet.
    """
    target = 'units' if group['scope'] == 'category' else 'sales'
    unit = 'units' if target == 'units' else 'brl'
    predicted_field, actual_field = 'predicted_' + target, 'actual_' + target
    history = dict(zip(group['dates'], group['values']))
    for kind in ['metrics', 'predictions', 'future']:
        result[kind] = [r for r in result[kind] if r['model'] != 'Naive']
    by_model = {name: {r['week_start']: r for r in result['predictions'] if r['model'] == name}
                for name in ['RF', 'Prophet']}
    baseline = [{**r, 'model': 'Naive', predicted_field: history[r['forecast_origin']], 'status': 'ok'}
                for r in by_model['RF'].values()]
    common = [r for r in baseline if all(
        by_model[name][r['week_start']][predicted_field] is not None
        and math.isfinite(by_model[name][r['week_start']][predicted_field])
        for name in by_model)]
    errors = [abs(r[actual_field] - r[predicted_field]) for r in common]
    total = sum(abs(r[actual_field]) for r in common)
    reference = result['metrics'][0]
    if len(common) != reference['scored_weeks']:
        raise ValueError('Naive scoring weeks do not match trained-model scoring weeks.')
    result['metrics'].append({**reference, 'model': 'Naive',
        'successful_weeks': len(baseline), 'constant_rounds': 0,
        'mae_' + unit: sum(errors) / len(errors) if errors else None,
        'rmse_' + unit: math.sqrt(sum(e*e for e in errors) / len(errors)) if errors else None,
        'wape_pct': 100 * sum(errors) / total if total else None,
        'status': 'ok' if errors else 'failed', 'message': ''})
    result['predictions'].extend(baseline)
    result['future'].extend([{**r, 'model': 'Naive', predicted_field: group['values'][-1],
                              'status': 'ok', 'message': ''}
                             for r in result['future'] if r['model'] == 'RF'])


def build_report(data, output):
    # All data/CSS/JS embedded: no fetch(), CDNs, server, SQL, or Python needed to view.
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False)
    for char, escaped in [('<', r'\u003c'), ('>', r'\u003e'), ('&', r'\u0026')]:
        payload = payload.replace(char, escaped)
    atomic_text(output, HTML.replace('__REPORT_DATA__', payload))
    print(f'HTML saved: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-only', action='store_true', help='Rebuild saved HTML; no SQL/training')
    parser.add_argument('--refresh', action='store_true', help='Retrain instead of using matching checkpoints')
    parser.add_argument('--workers', type=int, default=WORKERS)
    parser.add_argument('--sample', action='store_true', help='All sales groups + 3 category groups; separate sample outputs')
    args = parser.parse_args()
    if args.report_only and args.refresh:
        parser.error('--report-only and --refresh cannot be combined')
    if args.workers < 1:
        parser.error('--workers must be positive')
    output = REPORT_FILE.with_name('rf_prophet_sample.html') if args.sample else REPORT_FILE
    saved = RESULTS_DIR / ('sample_results.json' if args.sample else 'results.json')
    if args.report_only:
        if not saved.exists():
            parser.error(f'No saved results at {saved}. Run training first.')
        build_report(json.loads(saved.read_text(encoding='utf-8')), output)
        return
    if not 0 < INTERVAL_WIDTH < 1 or UNCERTAINTY_SAMPLES < 100:
        parser.error('Use an interval width between 0 and 1 and at least 100 samples.')
    initialize_models()
    groups = read_groups()
    if args.sample:
        chosen = {('Sudeste', 'papelaria'), ('Centro-Oeste', 'agro_industria_e_comercio'),
                  ('Nordeste', 'portateis_cozinha_e_preparadores_de_alimentos')}
        groups = [g for g in groups if g['scope'] != 'category' or (g['region'], g['category']) in chosen]
    import importlib.metadata
    settings = {'initial': INITIAL_TRAIN_WEEKS, 'horizon': HORIZON_WEEKS, 'future': FUTURE_WEEKS,
                'exclude_last_week': EXCLUDE_LAST_WEEK,
                'interval_width': INTERVAL_WIDTH, 'uncertainty_samples': UNCERTAINTY_SAMPLES,
                'packages': {p: importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','prophet']},
                'model_code': hashlib.sha256(''.join(inspect.getsource(f) for f in
                    [forecast_rf, month_inputs, forecast_prophet, predict_checked, evaluate_group]).encode()).hexdigest()}
    cache_dir = RESULTS_DIR / 'checkpoints'
    data = {'generated': datetime.now(timezone.utc).isoformat(), 'settings': settings,
            'sample': args.sample, 'sources': [], 'warnings': ['Sample report: only three category combinations.'] if args.sample else [],
            'scopes': {s: {'metrics': [], 'predictions': [], 'future': []} for s in ['global','region','category']},
            'diagnostics': []}
    pending, completed = [], 0
    def collect(group, result):
        add_naive_baseline(group, result)
        for key in ['metrics','predictions','future']:
            data['scopes'][group['scope']][key].extend(result[key])
        data['diagnostics'].extend(result['diagnostics'])
    for group in groups:
        fingerprint = hashlib.sha256(json.dumps([group, settings], sort_keys=True).encode()).hexdigest()
        path = cache_dir / (fingerprint + '.json')
        cached = None
        if path.exists() and not args.refresh:
            try:
                cached = json.loads(path.read_text(encoding='utf-8'))
                if not all(k in cached for k in ['metrics','predictions','future','diagnostics']):
                    cached = None
            except (ValueError, OSError):
                pass
        if cached is not None:
            needs_baseline = not any(r["model"] == "Naive" for r in cached["metrics"])
            collect(group, cached)
            if needs_baseline:
                save_json(path, cached)
            completed += 1
        else:
            pending.append((group, path))
    print(f'{len(groups)} groups: {completed} cached, {len(pending)} to compute; {args.workers} workers.', flush=True)
    # Each completed group is committed immediately. Reruns reuse matching data/settings/model code.
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_models) as pool:
        futures = {pool.submit(evaluate_group, (g, settings)): (g, p) for g, p in pending}
        for task in as_completed(futures):
            group, path = futures[task]
            result = task.result()
            collect(group, result)
            save_json(path, result)
            completed += 1
            print(f'[{completed}/{len(groups)}] {group["scope"]}: {group["region"]} {group["category"]}', flush=True)
    for scope in data['scopes'].values():
        for rows in scope.values():
            rows.sort(key=lambda r: (r['region'], r['category'], r['model'], r.get('week_start','')))
    # Publish final results/report only after every group finishes, retaining any older complete report meanwhile.
    save_json(saved, data)
    build_report(data, output)
    failures = [d for d in data['diagnostics'] if d['status']=='failed']
    print(f'Results saved: {saved}\nFailed model fits: {len(failures)}. Open the HTML directly any time.', flush=True)


HTML = r'''<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark"><title>OLIST · Naive, RF &amp; Prophet</title>
<style>
:root{--ink:#172f3c;--muted:#607580;--line:#dce5e6;--teal:#057c75;--blue:#366cc0;--orange:#c66b28;--bg:#f3f6f4;--card:#fff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 'Segoe UI',Arial,sans-serif}button,select{font:inherit}button,select,summary{cursor:pointer}button:focus-visible,select:focus-visible,summary:focus-visible,a:focus-visible{outline:3px solid #ce902e;outline-offset:3px}header{background:#132f3b;color:white;padding:30px max(24px,calc((100vw - 1320px)/2));border-bottom:4px solid #14a397}.topline{display:flex;justify-content:space-between;align-items:center;gap:15px}.brand{letter-spacing:.18em;font-size:12px;font-weight:700;color:#9ddad1}.badge{font-size:12px;border:1px solid #52717d;border-radius:30px;padding:5px 12px;color:#d9ebe9}h1{font-size:clamp(26px,3vw,39px);line-height:1.15;margin:17px 0 9px;letter-spacing:-.035em}header p{margin:0;color:#bed0d5;max-width:850px}.shell{max-width:1368px;margin:auto;padding:24px}.tabs{display:flex;gap:8px;padding:5px;background:#e5ece9;border-radius:13px;width:fit-content;max-width:100%;margin-bottom:24px}.tab{border:0;border-radius:9px;padding:11px 20px;background:transparent;color:#465f69;font-weight:650}.tab[aria-selected=true]{background:#fff;color:#086e68;box-shadow:0 2px 5px #19352b12}.tab small{font-weight:400;font-size:12px;display:block}.section-head{display:flex;justify-content:space-between;align-items:start;gap:12px;margin-bottom:18px}h2{font-size:24px;letter-spacing:-.02em;margin:0}h3{font-size:17px;margin:0 0 5px}.sub,.muted{color:var(--muted)}.sub{font-size:13px;margin:4px 0}.filters{display:flex;gap:16px;flex-wrap:wrap;background:white;border:1px solid var(--line);border-radius:12px;padding:17px 20px;margin-bottom:20px}label{display:flex;flex-direction:column;gap:6px;font-size:12px;font-weight:650;letter-spacing:.02em}select{background:#f9fbfa;border:1px solid #c9d6d6;padding:9px 32px 9px 11px;border-radius:7px;color:var(--ink);min-width:170px;max-width:380px}option{font-weight:400}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:22px}.card,.panel{background:white;border:1px solid var(--line);border-radius:12px}.card{padding:19px 21px;border-top:3px solid #b1d1c8}.eyebrow{text-transform:uppercase;letter-spacing:.1em;color:var(--muted);font-size:11px;font-weight:700}.big{font-size:26px;font-weight:700;letter-spacing:-.03em;margin:6px 0 3px}.card p{font-size:12px;margin:0;color:var(--muted)}.grid{display:grid;grid-template-columns:1fr 1.1fr;gap:20px}.panel{padding:21px;margin-bottom:20px;min-width:0}.panel-top{display:flex;justify-content:space-between;gap:10px;align-items:start}.panel-top select{min-width:110px;font-size:13px}.chart{min-height:270px;margin-top:16px}.chart svg{width:100%;height:auto;display:block}.chart svg text{font-family:'Segoe UI',Arial,sans-serif}.legend{display:flex;flex-wrap:wrap;gap:16px;font-size:12px;color:var(--muted);margin:8px 0}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}.table-wrap{overflow:auto;margin-top:17px}table{border-collapse:collapse;width:100%;font-size:13px}th{text-align:left;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.045em;border-bottom:1px solid var(--line);padding:10px 12px;white-space:nowrap}td{border-bottom:1px solid #edf1ef;padding:12px;white-space:nowrap}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}tr:last-child td{border-bottom:0}.pill{display:inline-block;padding:3px 8px;border-radius:20px;font-size:11px;background:#e6f3eb;color:#246648}.pill.bad{background:#fff0df;color:#955319}.note{font-size:12px;color:var(--muted);padding:11px 14px;background:#f1f6f3;border-left:3px solid #83ad9c;margin-top:12px}.notice{border-left:3px solid #d6a052;background:#fff6e7;color:#77511d;padding:12px 16px;border-radius:5px;font-size:13px;margin-bottom:16px}.empty{padding:38px 15px;text-align:center;color:var(--muted);font-size:14px;background:#f7f9f8;border-radius:8px}.section-divider{border-top:1px solid #cfdad8;padding-top:25px;margin-top:10px}details{font-size:13px;margin-top:14px}summary{font-weight:600}details pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f7f9f8;padding:12px;font-size:12px}footer{color:var(--muted);font-size:12px;padding:15px 0 35px}.sources td{white-space:normal;overflow-wrap:anywhere}#futureTable td{font-variant-numeric:tabular-nums}.hidden{display:none!important}.timestamp{font-size:12px;color:#b6cbd0}button{transition:background .15s}.selected-name{font-size:13px;font-weight:600;color:var(--teal);margin:6px 0;overflow-wrap:anywhere}
@media(max-width:850px){.grid{grid-template-columns:1fr}.shell{padding:16px}.cards{gap:10px}.card{padding:15px}.big{font-size:21px}.topline{align-items:start}.timestamp{display:none}.tab{padding:10px 14px}.section-head{flex-direction:column}select{max-width:100%}}@media(max-width:540px){.cards{grid-template-columns:1fr}.filters{flex-direction:column}select{width:100%}.tabs{width:100%}.tab{flex:1;padding:9px 7px;font-size:13px}.panel{padding:15px}.badge{display:none}header{padding:25px 18px}}
@media print{header{background:white;color:black}.tabs,.filters{display:none}.grid{display:block}.panel{break-inside:avoid}.chart{min-height:0}header p{color:#555}}

/* Quiet dark presentation */
:root{--ink:#e4eaf4;--muted:#99a6ba;--line:#293241;--teal:#6ad4b9;--bg:#10141c;--card:#171d28;color-scheme:dark}
body{font-size:14px;background:var(--bg)}.shell{max-width:1240px;padding:32px 28px}.tabs{background:transparent;padding:0;border-bottom:1px solid var(--line);border-radius:0;width:100%;gap:28px;margin-bottom:30px}.tab{padding:12px 0 16px;border-radius:0;color:var(--muted);font-size:15px}.tab[aria-selected=true]{background:transparent;color:var(--ink);box-shadow:none;border-bottom:2px solid var(--teal)}.tab small{display:none}.section-head{margin-bottom:20px}h2{font-size:24px;font-weight:600}h3{font-size:16px;font-weight:600}.filters{background:transparent;border:0;padding:0;gap:20px;margin-bottom:24px}select{background:#1b2330;border:1px solid #364155;color:var(--ink);border-radius:7px;padding:10px 32px 10px 12px}.cards{gap:0;border:1px solid var(--line);border-radius:10px;background:var(--card);margin-bottom:28px}.card{background:transparent;border:0;border-radius:0;padding:22px 26px}.card+.card{border-left:1px solid var(--line)}.big{font-size:26px;font-weight:600}.eyebrow{font-size:10px;letter-spacing:.08em}.panel{background:var(--card);border-color:var(--line);padding:24px;border-radius:10px;margin-bottom:26px}.grid{gap:24px;grid-template-columns:1fr 1.15fr}.chart{min-height:0;margin-top:24px}.metric-row{margin:0 0 22px}.metric-row:last-child{margin-bottom:6px}.metric-line{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:9px;font-size:13px}.metric-line strong{font-variant-numeric:tabular-nums;font-weight:600}.track{height:7px;background:#242d3c;border-radius:8px;overflow:hidden}.track>div{height:100%;border-radius:8px}.best-tag{font-size:10px;color:var(--teal);margin-left:9px;font-weight:400}.table-wrap{margin-top:14px}th{padding:12px 10px;font-size:10px}td{padding:17px 10px;border-color:var(--line)}.section-divider{border-top:0;padding-top:8px}.selected-name{color:var(--muted);font-weight:400}.empty{background:#1b2330}.legend{gap:22px}.dot{width:8px;height:8px}.sub:empty{display:none}#snapshot,#evaluationNotice,#futureNotice,#scoreNote,#metricHelp,#diagnostics,#futureDiagnostics,#actualDetail{display:none!important}details pre{background:#1b2330}summary{color:var(--muted);font-weight:400}.notice{background:#242b37;color:#d5dfed;border-color:#8995ab}
@media(max-width:850px){.grid{grid-template-columns:1fr}.shell{padding:24px 18px}.tabs{gap:22px}.card{padding:18px}.big{font-size:22px}}@media(max-width:540px){.cards{grid-template-columns:1fr}.card+.card{border-left:0;border-top:1px solid var(--line)}.tabs{gap:12px}.tab{font-size:13px}.panel{padding:18px}.metric-line{font-size:12px}.best-tag{display:none}}
#evaluationNotice,#futureNotice{display:block!important} .notice{font-size:12px;padding:9px 12px;margin-bottom:18px} .demand-pie{display:grid;grid-template-columns:minmax(220px,360px) 1fr;align-items:center;gap:36px}.demand-pie .metric-line{margin-bottom:16px}.demand-pie .metric-line span{overflow-wrap:anywhere}@media(max-width:650px){.demand-pie{grid-template-columns:1fr;gap:16px}.demand-pie>div:first-child{max-width:320px;width:100%;margin:auto}} .demand-donut{overflow-x:auto}.demand-donut svg{min-width:760px}.demand-slice{transition:filter .15s}.demand-slice:hover{filter:brightness(1.12)} </style></head>
<body>
<main class="shell">
<nav class="tabs" role="tablist" aria-label="Analysis scope">
<button class="tab" id="tab-global" role="tab" aria-controls="analysis" aria-selected="true" data-scope="global">Global<small>Sales value</small></button>
<button class="tab" id="tab-region" role="tab" aria-controls="analysis" aria-selected="false" data-scope="region" tabindex="-1">Regions<small>Sales value</small></button>
<button class="tab" id="tab-category" role="tab" aria-controls="analysis" aria-selected="false" data-scope="category" tabindex="-1">Region + Category<small>Units sold</small></button></nav>
<div id="analysis" role="tabpanel" aria-labelledby="tab-global">
<div class="section-head"><div><h2 id="scopeTitle"></h2><p class="sub" id="scopeSubtitle"></p></div><span class="muted" id="snapshot" style="font-size:12px"></span></div>
<div class="filters"><label id="regionField">REGION<select id="regionSelect"></select></label><label id="categoryField">CATEGORY<select id="categorySelect"></select></label><label id="evaluationField">EVALUATION RUN<select id="evaluationSelect"></select></label></div>
<div id="loadWarnings"></div><div id="evaluationNotice"></div>
<div class="cards" aria-live="polite"><div class="card"><div class="eyebrow">Lowest average error</div><div class="big" id="bestModel">—</div><p id="bestDetail"></p></div><div class="card"><div class="eyebrow" id="totalLabel">Actual sales</div><div class="big" id="actualTotal">—</div><p id="actualDetail"></p></div><div class="card"><div class="eyebrow">Evaluation window</div><div class="big" id="periodLabel">—</div><p id="periodDetail"></p></div></div>
<div class="grid"><section class="panel"><div class="panel-top"><div><h3>Forecast error</h3><p class="sub">Lower is better</p></div><label aria-label="Error metric"><select id="metricSelect"><option value="mae">MAE</option><option value="rmse">RMSE</option><option value="wape">WAPE</option></select></label></div><div class="chart" id="metricChart"></div><p class="sub" id="metricHelp"></p></section>
<section class="panel"><h3>Model comparison</h3><p class="sub"></p><div class="table-wrap" id="scoreTable"></div><div class="note" id="scoreNote"></div><details id="diagnostics"><summary>Fit diagnostics</summary><pre id="diagnosticText"></pre></details></section></div>
<section class="panel"><h3>Actual vs predicted</h3><p class="sub" id="evaluationCaption"></p><div class="legend" id="evaluationLegend"></div><div class="chart" id="evaluationChart"></div><details><summary>View weekly evaluation values</summary><div class="table-wrap" id="evaluationTable"></div></details></section>
<section class="panel"><h3>Prophet uncertainty</h3><p class="sub" id="uncertaintyCaption"></p><div class="legend" id="uncertaintyLegend"></div><div class="chart" id="uncertaintyChart"></div><p class="sub" id="uncertaintySummary" aria-live="polite"></p></section>
<div class="section-head section-divider"><div><h2>Next 4 weeks</h2><p class="sub"></p></div><label id="futureField">FUTURE FORECAST RUN<select id="futureSelect"></select></label></div>
<div id="futureNotice"></div><section class="panel"><p class="selected-name" id="futureCaption"></p><div class="legend" id="futureLegend"></div><div class="chart" id="futureChart"></div><div class="table-wrap" id="futureTable"></div><details id="futureDiagnostics"><summary>Future fit diagnostics</summary><pre id="futureDiagnosticText"></pre></details></section>
</div>
<section class="panel hidden" id="categoryDemandPanel"><div class="panel-top"><div><h3>Top 10 categories by expected demand</h3><p class="sub" id="categoryDemandCaption"></p></div><label>MODEL<select id="categoryDemandModel"><option>Prophet</option><option>RF</option></select></label></div><div id="categoryDemandNotice"></div><div class="chart" id="categoryDemandChart"></div><details><summary>View weekly category forecasts</summary><div class="table-wrap" id="categoryDemandTable"></div></details></section>
<div hidden id="sources"></div>
</main>
<script id="report-data" type="application/json">__REPORT_DATA__</script>
<script>
'use strict';
const DATA=JSON.parse(document.getElementById('report-data').textContent);
const $=id=>document.getElementById(id), esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const colors={Actual:'#edf2f8',Naive:'#8995ab',RF:'#6ad4b9',ARIMAX:'#85aaff',SARIMAX:'#9259b5',Prophet:'#e5ad68'};
const state={scope:'global'}, finite=v=>typeof v==='number'&&Number.isFinite(v);
const unique=a=>[...new Set(a)].sort(), num=v=>finite(v)?v.toLocaleString('en-US',{maximumFractionDigits:2,minimumFractionDigits:2}):'—';
const date=d=>d?new Date(d+'T12:00:00').toLocaleDateString('en-GB',{day:'2-digit',month:'short',year:'numeric'}):'—';
const shortDate=d=>new Date(d+'T12:00:00').toLocaleDateString('en-GB',{day:'2-digit',month:'short'});
const pretty=c=>c==='unknown_category'?'Unknown category':c.replaceAll('_',' ');
const isUnits=()=>state.scope==='category', amount=v=>finite(v)?(isUnits()?num(v)+' units':'R$ '+num(v)):'—';
const bool=v=>String(v).toLowerCase()==='true'||String(v)==='1';
const evalKey=r=>JSON.stringify([r.test_weeks,r.train_end,r.test_start,r.test_end,bool(r.exclude_last_week)]);
const futureKey=r=>JSON.stringify([r.forecast_weeks,r.train_end,bool(r.exclude_last_week)]);
function options(id,items,preferred){const el=$(id), old=preferred??el.value;el.replaceChildren();for(const [value,label] of items){const op=document.createElement('option');op.value=value;op.textContent=label;el.append(op)}if(items.some(i=>i[0]===old))el.value=old;el.disabled=!items.length;}
function bundle(){return DATA.scopes[state.scope]}
function groupRows(rows){return rows.filter(r=>(state.scope==='global'||r.region===$('regionSelect').value)&&(state.scope!=='category'||r.category===$('categorySelect').value))}
function updateFilters(){
 const all=[...bundle().metrics,...bundle().future];
 $('regionField').classList.toggle('hidden',state.scope==='global');$('categoryField').classList.toggle('hidden',!isUnits());
 options('regionSelect',unique(all.map(r=>r.region)).map(v=>[v,v]),$('regionSelect').value||'Sudeste');
 updateCategories();
}
function updateCategories(){
 const all=[...bundle().metrics,...bundle().future].filter(r=>r.region===$('regionSelect').value);
 options('categorySelect',unique(all.map(r=>r.category).filter(Boolean)).map(v=>[v,pretty(v)]),$('categorySelect').value||'papelaria');updateRuns();
}
function updateRuns(){
 const evals=new Map(groupRows(bundle().metrics).map(r=>[evalKey(r),r]));
 const entries=[...evals].sort((a,b)=>b[1].train_end.localeCompare(a[1].train_end)||a[1].test_weeks-b[1].test_weeks);
 options('evaluationSelect',entries.map(([k,r])=>[k,`${r.test_weeks} weeks · ${date(r.test_start)} – ${date(r.test_end)}`]));
 const future=new Map(groupRows(bundle().future).map(r=>[futureKey(r),r]));
 options('futureSelect',[...future].sort((a,b)=>b[1].train_end.localeCompare(a[1].train_end)).map(([k,r])=>[k,`${r.forecast_weeks} weeks · cutoff ${date(r.train_end)}`]));$('evaluationField').classList.toggle('hidden',entries.length<=1);$('futureField').classList.toggle('hidden',future.size<=1);document.querySelector('.filters').classList.toggle('hidden',state.scope==='global'&&entries.length<=1);render();
}
function note(id,text){$(id).innerHTML=text?`<div class="notice">${esc(text)}</div>`:''}
function empty(id,text){$(id).innerHTML=`<div class="empty">${esc(text)}</div>`}
function legend(id,models){$(id).innerHTML=models.map(m=>`<span><i class="dot" style="background:${colors[m]||'#777'}"></i>${esc(m)}</span>`).join('')}
function metricField(){const m=$('metricSelect').value;return m==='wape'?'wape_pct':m+'_'+(isUnits()?'units':'brl')}
function barChart(rows){
 const field=metricField(),data=rows.filter(r=>r.status==='ok'&&finite(r[field])).sort((a,b)=>a[field]-b[field]);
 if(!data.length){empty('metricChart',field==='wape_pct'?'WAPE unavailable for this period':'No results available');return}
 const max=Math.max(...data.map(r=>r[field]),1);
 $('metricChart').innerHTML=data.map((r,i)=>`<div class="metric-row"><div class="metric-line"><span><i class="dot" style="background:${colors[r.model]}"></i>${esc(r.model)}${Math.abs(r[field]-data[0][field])<1e-8*Math.max(1,Math.abs(data[0][field]))?'<small class="best-tag">Lowest error</small>':''}</span><strong>${field==='wape_pct'?num(r[field])+'%':amount(r[field])}</strong></div><div class="track"><div style="width:${100*r[field]/max}%;background:${colors[r.model]}"></div></div></div>`).join('');
}
function lineChart(id,dates,series,future=false,band=null){
 const vals=series.flatMap(s=>s.values.filter(finite)).concat(band?[...band.lower,...band.upper].filter(finite):[]);if(!dates.length||!vals.length){empty(id,'No valid predictions available for this selection.');return}
 const W=1150,H=300,L=78,R=28,T=20,B=45,max=Math.max(...vals,1)*1.12;
 const x=i=>L+(dates.length===1?(W-L-R)/2:(W-L-R)*i/(dates.length-1));const y=v=>H-B-(H-T-B)*v/max;
 let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${future?'Future forecasts':'Actual and predicted values'} by week">`;
 for(let i=0;i<=4;i++){const val=max*i/4,yy=y(val);svg+=`<line x1="${L}" x2="${W-R}" y1="${yy}" y2="${yy}" stroke="#293242"/><text x="${L-10}" y="${yy+4}" text-anchor="end" font-size="11" fill="#9aa8bd">${esc(val.toLocaleString('en-US',{notation:'compact',maximumFractionDigits:1}))}</text>`}
 const stride=Math.max(1,Math.ceil(dates.length/9));dates.forEach((d,i)=>{if((i%stride===0&&i<dates.length-1-Math.floor(stride/2))||i===dates.length-1)svg+=`<text x="${x(i)}" y="${H-16}" text-anchor="middle" font-size="11" fill="#9aa8bd">${esc(shortDate(d))}</text>`});
 if(band){let segment=[];const draw=()=>{if(!segment.length)return;const upper=segment.map(i=>`${x(i)},${y(band.upper[i])}`),lower=[...segment].reverse().map(i=>`${x(i)},${y(band.lower[i])}`);if(segment.length===1){const i=segment[0];svg+=`<rect x="${x(i)-3}" y="${y(band.upper[i])}" width="6" height="${Math.max(1,y(band.lower[i])-y(band.upper[i]))}" fill="#e5ad68" opacity="0.23"/>`}else svg+=`<polygon points="${[...upper,...lower].join(' ')}" fill="#e5ad68" opacity="0.23"/>`;segment=[]};dates.forEach((d,i)=>{if(finite(band.lower[i])&&finite(band.upper[i]))segment.push(i);else draw()});draw()}
 series.forEach(s=>{let path='',started=false;s.values.forEach((v,i)=>{if(!finite(v)){started=false;return}path+=(started?' L':' M')+x(i)+','+y(v);started=true});svg+=`<path d="${path}" fill="none" stroke="${colors[s.name]||'#777'}" stroke-width="${s.name==='Actual'?3.4:2.2}" ${future?'stroke-dasharray="7 4"':''}/>`;
 s.values.forEach((v,i)=>{if(finite(v))svg+=`<circle cx="${x(i)}" cy="${y(v)}" r="${dates.length>30?2.5:4}" fill="${colors[s.name]||'#777'}" stroke="#171d28" stroke-width="1"><title>${esc(s.name)} · ${esc(date(dates[i]))}: ${esc(amount(v))}</title></circle>`})});
 $(id).innerHTML=svg+'</svg>';
}
function valuesTable(id,dates,series){if(!dates.length){empty(id,'No weekly values available.');return}$(id).innerHTML=`<table><thead><tr><th>Week starting</th>${series.map(s=>`<th class="num">${esc(s.name)}</th>`).join('')}</tr></thead><tbody>${dates.map((d,i)=>`<tr><td>${esc(date(d))}</td>${series.map(s=>`<td class="num">${esc(num(s.values[i]))}</td>`).join('')}</tr>`).join('')}</tbody></table>`}
function renderEvaluation(){
 const rows=groupRows(bundle().metrics).filter(r=>evalKey(r)===$('evaluationSelect').value), ref=rows[0];
 $('totalLabel').textContent=isUnits()?'Total test units':'Total test sales';
 if(!ref){renderUncertainty([],[],isUnits()?'actual_units':'actual_sales',isUnits()?'predicted_units':'predicted_sales');['bestModel','actualTotal','periodLabel'].forEach(id=>$(id).textContent='—');['bestDetail','actualDetail','periodDetail','evaluationCaption','scoreNote'].forEach(id=>$(id).textContent='');['metricChart','scoreTable','evaluationChart','evaluationTable'].forEach(id=>empty(id,'No evaluation data found.'));$('evaluationLegend').innerHTML='';$('diagnostics').classList.add('hidden');note('evaluationNotice','');return}
 const mf=isUnits()?'mae_units':'mae_brl',rf=isUnits()?'rmse_units':'rmse_brl',tf=isUnits()?'actual_units_total':'actual_sales_total';
 const valid=rows.filter(r=>r.status==='ok'&&finite(r[mf])).sort((a,b)=>a[mf]-b[mf]);
 const winner=valid[0],ties=winner?valid.filter(r=>Math.abs(r[mf]-winner[mf])<=1e-8*Math.max(1,Math.abs(winner[mf]))):[];
 $('bestModel').textContent=ties.map(r=>r.model).join(' / ')||'No valid fit';
 const baseline=valid.find(r=>r.model==='Naive'),improvement=winner&&baseline&&baseline[mf]>0?100*(1-winner[mf]/baseline[mf]):null;
 $('bestDetail').textContent=winner?`${amount(winner[mf])} MAE${improvement!==null?' · '+improvement.toFixed(1)+'% below naive':''}`:'Inspect fit diagnostics.';
 $('actualTotal').textContent=amount(ref[tf]);$('actualDetail').textContent=isUnits()?'Delivered item rows in the test period':'Delivered-item prices; excludes freight and costs';
 $('periodLabel').textContent=`${ref.test_weeks} weeks`;$('periodDetail').textContent=`${date(ref.test_start)} – ${date(ref.test_end)}`;
 const issues=[];
 if(ref.scored_weeks<ref.test_weeks)issues.push(`Metrics use ${ref.scored_weeks} / ${ref.test_weeks} shared successful weeks.`);
 if(ref[tf]===0)issues.push('No test units sold · WAPE unavailable.');
 if(rows.some(r=>r.constant_rounds>0))issues.push('Some rounds used a constant forecast because training history had no variation.');
 note('evaluationNotice',issues.join(' '));
 barChart(rows);$('metricHelp').textContent=({mae:'MAE: average absolute miss, in '+(isUnits()?'units.':'Brazilian reais.'),rmse:'RMSE: error in '+(isUnits()?'units':'Brazilian reais')+', with more weight on large misses.',wape:'WAPE: total absolute error ÷ total actual demand × 100. Can exceed 100%.'})[$('metricSelect').value];
 $('scoreTable').innerHTML=`<table><thead><tr><th>Model</th><th class="num">MAE</th><th class="num">RMSE</th><th class="num">WAPE</th></tr></thead><tbody>${rows.map(r=>`<tr><td><span class="dot" style="background:${colors[r.model]||'#777'}"></span>${esc(r.model)}</td><td class="num">${r.status==='ok'?num(r[mf]):'—'}</td><td class="num">${r.status==='ok'?num(r[rf]):'—'}</td><td class="num">${r.status==='ok'&&finite(r.wape_pct)?num(r.wape_pct)+'%':'—'}</td></tr>`).join('')}</tbody></table>`;
 $('scoreNote').textContent=`MAE/RMSE: ${isUnits()?'units':'BRL'}. “—” means unavailable, never zero. Rank applies only to this selection and evaluation run.`;
 const msgs=rows.filter(r=>r.message||r.status!=='ok').map(r=>`${r.model}: ${r.message||r.status}`);$('diagnostics').classList.toggle('hidden',!msgs.length);$('diagnosticText').textContent=msgs.join('\n\n');
 const pred=groupRows(bundle().predictions).filter(r=>r.train_end===ref.train_end&&r.test_weeks===ref.test_weeks&&r.week_start>=ref.test_start&&r.week_start<=ref.test_end&&(!('exclude_last_week'in r)||bool(r.exclude_last_week)===bool(ref.exclude_last_week)));
 const dates=unique(pred.map(r=>r.week_start)),af=isUnits()?'actual_units':'actual_sales',pf=isUnits()?'predicted_units':'predicted_sales';
 const series=[{name:'Actual',values:dates.map(d=>pred.find(r=>r.week_start===d)?.[af]??null)},...unique(pred.map(r=>r.model)).map(name=>({name,values:dates.map(d=>pred.find(r=>r.week_start===d&&r.model===name)?.[pf]??null)}))];
 $('evaluationCaption').textContent=`${date(ref.test_start)} – ${date(ref.test_end)} · ${ref.horizon_weeks}-week walk-forward · ${isUnits()?'Units':'BRL'}`;
 renderUncertainty(pred,dates,af,pf);
 legend('evaluationLegend',series.map(s=>s.name));lineChart('evaluationChart',dates,series);valuesTable('evaluationTable',dates,series);
}
function renderUncertainty(pred,dates,af,pf){
 const rows=pred.filter(r=>r.model==='Prophet'), lookup=new Map(rows.map(r=>[r.week_start,r]));
 const valid=rows.filter(r=>finite(r.lower)&&finite(r.upper)&&finite(r[af]));
 const level=DATA.settings.interval_width;
 $('uncertaintyLegend').innerHTML='';$('uncertaintySummary').textContent='';
 $('uncertaintyCaption').textContent='Historical walk-forward forecasts · '+(isUnits()?'Units':'BRL');
 if(!valid.length||!finite(level)){empty('uncertaintyChart','No estimated intervals for this selection. Older results need a training run; constant-history forecasts have no fitted Prophet interval.');return}
 const lower=dates.map(d=>lookup.get(d)?.lower??null),upper=dates.map(d=>lookup.get(d)?.upper??null);
 const series=[{name:'Actual',values:dates.map(d=>lookup.get(d)?.[af]??null)},{name:'Prophet',values:dates.map(d=>lookup.get(d)?.[pf]??null)}];
 legend('uncertaintyLegend',['Actual','Prophet']);$('uncertaintyLegend').innerHTML+=`<span><i class="dot" style="background:#e5ad68;opacity:.5"></i>${num(level*100)}% prediction interval</span>`;
 lineChart('uncertaintyChart',dates,series,false,{lower,upper});
 const hits=valid.filter(r=>r[af]>=r.lower&&r[af]<=r.upper).length,coverage=100*hits/valid.length;
 const width=valid.reduce((sum,r)=>sum+r.upper-r.lower,0)/valid.length;
 const interpretation=coverage<level*100?'Observed coverage was below the nominal level, suggesting underestimated uncertainty in this evaluation.':'Observed coverage met or exceeded the nominal level; this does not guarantee future coverage.';
 $('uncertaintySummary').textContent=`The nominal ${num(level*100)}% interval contained actual values in ${hits} of ${valid.length} scored weeks (${num(coverage)}%). Average band width: ${amount(width)}. ${interpretation}`+(valid.length<rows.length?` Bounds unavailable for ${rows.length-valid.length} weeks (constant-history fallback or failed fit); those weeks are excluded from interval coverage.`:'');
}
function renderFuture(){
 const rows=groupRows(bundle().future).filter(r=>futureKey(r)===$('futureSelect').value),ref=rows[0];
 if(!ref){$('futureCaption').textContent='';$('futureLegend').innerHTML='';empty('futureChart','No future forecast file found for this selection.');empty('futureTable','No future values available.');note('futureNotice','');$('futureDiagnostics').classList.add('hidden');return}
 const dates=unique(rows.map(r=>r.week_start)),pf=isUnits()?'predicted_units':'predicted_sales';
 const names=unique(rows.map(r=>r.model)),series=names.map(name=>({name,values:dates.map(d=>{const row=rows.find(r=>r.week_start===d&&r.model===name);return row?.status==='ok'?row[pf]:null})}));
 $('futureCaption').textContent=`${date(dates[0])} – ${date(dates.at(-1))} · ${isUnits()?'Units':'BRL'}`;document.querySelector('.section-divider h2').textContent=`Next ${ref.forecast_weeks} weeks`;
 let message=rows.some(r=>r.status!=='ok')?'Some forecasts are unavailable (—).':'';
 note('futureNotice',message);legend('futureLegend',names);lineChart('futureChart',dates,series,true);valuesTable('futureTable',dates,series);
 const msgs=unique(rows.filter(r=>r.message||r.status!=='ok').map(r=>`${r.model}: ${r.message||r.status}`));$('futureDiagnostics').classList.toggle('hidden',!msgs.length);$('futureDiagnosticText').textContent=msgs.join('\n\n');
}
function renderCategoryDemand(){
 $('categoryDemandPanel').classList.toggle('hidden',state.scope!=='region');
 if(state.scope!=='region')return;
 const region=$('regionSelect').value,model=$('categoryDemandModel').value;
 const regional=groupRows(bundle().future).filter(r=>futureKey(r)===$('futureSelect').value);
 const ref=regional[0],dates=unique(regional.map(r=>r.week_start));
 const clear=text=>{empty('categoryDemandChart',text);$('categoryDemandTable').innerHTML='';$('categoryDemandCaption').textContent='';note('categoryDemandNotice','')};
 if(!ref||dates.length!==ref.forecast_weeks){clear('No complete forecast horizon available.');return}
 const rows=DATA.scopes.category.future.filter(r=>r.region===region&&r.model===model&&futureKey(r)===futureKey(ref));
 const categories=unique([...DATA.scopes.category.metrics,...DATA.scopes.category.future].filter(r=>r.region===region).map(r=>r.category));
 const ranked=[];
 for(const category of categories){
  const part=rows.filter(r=>r.category===category);
  const weekly=dates.map(d=>part.filter(r=>r.week_start===d));
  if(!weekly.every(a=>a.length===1&&a[0].status==='ok'&&finite(a[0].predicted_units)&&a[0].predicted_units>=0))continue;
  const values=weekly.map(a=>a[0].predicted_units);
  ranked.push({category,values,total:values.reduce((a,b)=>a+b,0)});
 }
 ranked.sort((a,b)=>b.total-a.total||a.category.localeCompare(b.category));
 const top=ranked.slice(0,10),excluded=categories.length-ranked.length;
 $('categoryDemandCaption').textContent=`${region} · ${model} · ${date(dates[0])} – ${date(dates.at(-1))} · Ranked by total expected units over ${dates.length} weeks`;
 note('categoryDemandNotice',(excluded?`${excluded} categories excluded because a valid forecast was unavailable for one or more weeks. `:'')+(DATA.sample?'Sample report: ranking covers only the included categories.':''));
 if(!top.length){empty('categoryDemandChart','No categories have complete forecasts for this model and horizon.');$('categoryDemandTable').innerHTML='';return}
 const total=top.reduce((sum,r)=>sum+r.total,0);
 if(total<=0){empty('categoryDemandChart','All selected category forecasts are zero; no demand shares to display.')}else{
  const palette=['#6ad4b9','#85aaff','#e5ad68','#b69aed','#ef8e9e','#80c9e0','#c8cf7d','#f2bdcc','#8cbf98','#d69b79'];
  const cx=550,cy=370,radius=190,labels=[];
  const point=(a,r)=>[cx+r*Math.cos(a),cy+r*Math.sin(a)];
  let angle=-Math.PI/2;
  let svg='<svg viewBox="0 0 1100 740" role="img" aria-label="Top ten categories with forecast unit totals connected to their slices"><defs><filter id="pieShadow" x="-30%" y="-30%" width="160%" height="160%"><feDropShadow dx="0" dy="8" stdDeviation="10" flood-opacity="0.22"/></filter></defs><g filter="url(#pieShadow)">';
  top.forEach((r,i)=>{
   const share=r.total/total,end=angle+share*2*Math.PI,mid=(angle+end)/2;
   const start=point(angle,radius),finish=point(end,radius);
   const title=esc(pretty(r.category))+': '+num(r.total)+' forecast units';
   if(share>=1-1e-12)svg+=`<circle cx="${cx}" cy="${cy}" r="${radius}" fill="${palette[i]}"><title>${title}</title></circle>`;
   else if(share>0)svg+=`<path class="demand-slice" d="M${cx},${cy} L${start} A${radius},${radius} 0 ${share>0.5?1:0},1 ${finish} Z" fill="${palette[i]}" stroke="#171d28" stroke-width="3"><title>${title}</title></path>`;
   if(share>0)labels.push({r,i,share,mid,side:Math.cos(mid)>=0?1:-1,y:cy+Math.sin(mid)*210});
   angle=end;
  });
  svg+=`</g><circle cx="${cx}" cy="${cy}" r="103" fill="#171d28"/><circle cx="${cx}" cy="${cy}" r="98" fill="none" stroke="#293241"/>`;
  for(const side of [-1,1]){
   const group=labels.filter(l=>l.side===side).sort((a,b)=>a.y-b.y);
   group.forEach((l,i)=>l.y=Math.max(l.y,50,i?group[i-1].y+64:50));
   if(group.length&&group.at(-1).y>665){group.at(-1).y=665;for(let i=group.length-2;i>=0;i--)group[i].y=Math.min(group[i].y,group[i+1].y-64)}
  }
  labels.forEach(l=>{
   const start=point(l.mid,radius+4),bend=point(l.mid,radius+23),endX=l.side===1?790:310,textX=endX+l.side*12,anchor=l.side===1?'start':'end';
   const words=pretty(l.r.category).split(' '),lines=[''];
   for(const word of words){if((lines.at(-1)+' '+word).trim().length>29&&lines.at(-1))lines.push(word);else lines[lines.length-1]=(lines.at(-1)+' '+word).trim()}
   svg+=`<polyline points="${start} ${bend} ${endX-l.side*12},${l.y} ${endX},${l.y}" fill="none" stroke="${palette[l.i]}" stroke-opacity="0.7" stroke-width="1.5"/><circle cx="${start[0]}" cy="${start[1]}" r="3" fill="${palette[l.i]}"/><text x="${textX}" y="${l.y-8}" text-anchor="${anchor}" font-size="15" fill="#e4eaf4">${lines.map((line,j)=>`<tspan x="${textX}" dy="${j?18:0}">${esc(line)}</tspan>`).join('')}<tspan x="${textX}" dy="23" fill="${palette[l.i]}" font-weight="600" font-size="17">${num(l.r.total)} units</tspan></text>`;
  });
  svg+='</svg>';
  $('categoryDemandChart').innerHTML=`<div class="demand-donut">${svg}</div><p class="sub" style="text-align:center">Total predicted units over the forecast horizon. Slice sizes compare these top ten categories only.</p>`;

 }
 $('categoryDemandTable').innerHTML=`<table><thead><tr><th>Category</th>${dates.map(d=>`<th class="num">${esc(shortDate(d))}</th>`).join('')}<th class="num">Total units</th></tr></thead><tbody>${top.map(r=>`<tr><td>${esc(pretty(r.category))}</td>${r.values.map(v=>`<td class="num">${num(v)}</td>`).join('')}<td class="num"><strong>${num(r.total)}</strong></td></tr>`).join('')}</tbody></table>`;
}
function render(){
 const titles={global:['Global sales','All regions and product categories · sales value in BRL'],region:['Regional sales',$('regionSelect').value+' · all product categories · sales value in BRL'],category:['Category demand',$('regionSelect').value+' / '+pretty($('categorySelect').value)+' · delivered units']};
 $('scopeTitle').textContent=titles[state.scope][0];$('scopeSubtitle').textContent=titles[state.scope][1];renderEvaluation();renderFuture();renderCategoryDemand();
}
document.querySelectorAll('[data-scope]').forEach((button,index,buttons)=>{button.onclick=()=>{state.scope=button.dataset.scope;buttons.forEach(b=>{const active=b===button;b.setAttribute('aria-selected',active);b.tabIndex=active?0:-1});$('analysis').setAttribute('aria-labelledby',button.id);updateFilters()};button.onkeydown=e=>{if(['ArrowRight','ArrowLeft','Home','End'].includes(e.key)){e.preventDefault();let i=e.key==='Home'?0:e.key==='End'?buttons.length-1:(index+(e.key==='ArrowRight'?1:-1)+buttons.length)%buttons.length;buttons[i].click();buttons[i].focus()}}});
$('regionSelect').onchange=updateCategories;$('categorySelect').onchange=updateRuns;$('evaluationSelect').onchange=renderEvaluation;$('futureSelect').onchange=()=>{renderFuture();renderCategoryDemand()};$('categoryDemandModel').onchange=renderCategoryDemand;$('metricSelect').onchange=renderEvaluation;
$('snapshot').textContent='Built '+new Date(DATA.generated).toLocaleString();note('loadWarnings',DATA.warnings.join(' '));
$('sources').innerHTML=`<table><thead><tr><th>CSV snapshot</th><th class="num">Rows</th><th>Last modified</th></tr></thead><tbody>${DATA.sources.map(s=>`<tr><td>${esc(s.name)}</td><td class="num">${s.rows.toLocaleString()}</td><td>${esc(new Date(s.modified).toLocaleString())}</td></tr>`).join('')}</tbody></table>`;
updateFilters();
</script></body></html>'''

if __name__ == '__main__':
    main()
