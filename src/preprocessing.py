"""Loading, audit (section 3.1), cleaning (5.1), station IDs, matching to GEE (5.3.4), station retention,
and the leakage-free preprocessing: missing values (5.3.1), scaling and feature selection (5.3.2),
outliers (5.3.3) and the reusable Pipeline. Every learned step is fitted on a training fold only."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.feature_selection import RFE, SelectKBest, f_regression, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.config import (GAS, GEE_FILE, MAX_DIST, MIN_COVERAGE, MIN_OBS, PM25_MAX, PM25_MIN, PM_FILE, SEED,
                        STATION_TOL, TARGET)
from src.utils import log, savefig, show

PM_LABEL, GEE_LABEL = PM_FILE.name, GEE_FILE.name


# ---------------------------------------------------------------- loading and audit (section 3.1)
def load_raw():
    P = pd.read_csv(PM_FILE, parse_dates=["date"])
    G = pd.read_csv(GEE_FILE, parse_dates=["date", "datetime_utc"])
    return P, G


def dataset_description():
    """1.1 Source, license, units."""
    license_ = ("Other (specified in description) on Kaggle; built from OpenAQ (CC BY 4.0), "
                "Copernicus (open access) and NASA open data")
    info = pd.DataFrame({"item": ["Dataset", "Source", "License", "Files used", "Target", "Task"],
        "value": ["Air Quality - Mediterranean Basin (2016-2026)",
                  "https://www.kaggle.com/datasets/sarahameziane/air-quality-mediterranean-basin-20162026/",
                  license_, f"{PM_LABEL} (ground stations), {GEE_LABEL} (Google Earth Engine features)",
                  "pm25: daily surface PM2.5 concentration, ug/m3", "Regression / forecasting (not classification)"]})
    variables = pd.DataFrame([
        ["pm25", "Daily PM2.5 at a ground station (TARGET)", "ug/m3", "ground station"],
        ["AOD", "Aerosol optical depth (light blocked by airborne particles)", "unitless", "satellite"],
        ["NO2", "Nitrogen dioxide column", "mol/m2 (inferred from magnitude)", "Sentinel-5P"],
        ["CO", "Carbon monoxide column", "mol/m2 (inferred from magnitude)", "Sentinel-5P"],
        ["O3", "Ozone column", "mol/m2 (inferred from magnitude)", "Sentinel-5P"],
        ["temperature_celsius", "Air temperature", "degrees C", "reanalysis"],
        ["pressure_mb", "Surface pressure", "mb (= hPa)", "reanalysis"],
        ["wind_u", "Wind component, east-west", "m/s (inferred)", "reanalysis"],
        ["wind_v", "Wind component, north-south", "m/s (inferred)", "reanalysis"],
        ["sensor_name / lat / lon", "Station name and coordinates (group labels, NOT predictors)", "-", "ground station"],
        ["id, lat, lon (GEE file)", "ID and coordinates of the GEE grid point", "-", "GEE"]],
        columns=["variable", "meaning", "unit", "origin"])
    show(info)
    show(variables)


def _coverage(d, lat, lon, ident):
    return {"rows": len(d), "columns": d.shape[1], "first_date": d.date.min().date(), "last_date": d.date.max().date(),
            "distinct_dates": d.date.nunique(), "distinct_" + ident: d[ident].nunique(),
            "lat_min": round(d[lat].min(), 3), "lat_max": round(d[lat].max(), 3),
            "lon_min": round(d[lon].min(), 3), "lon_max": round(d[lon].max(), 3)}


def audit_size_and_coverage(P, G, rows):
    """1.2 Rows, columns, date range, spatial coverage."""
    cov = pd.DataFrame({PM_LABEL: _coverage(P, "latitude", "longitude", "sensor_name"),
                        GEE_LABEL: _coverage(G, "lat", "lon", "id")}).T
    show(cov)
    for f, r in cov.iterrows():
        for k, v in r.items():
            log(rows, "audit_coverage", f, k, v)


def audit_missing_and_duplicates(P, G, rows):
    """1.3 Missing values and duplicates (raw files)."""
    miss = pd.DataFrame({PM_LABEL + " missing %": P.isna().mean() * 100, GEE_LABEL + " missing %": G.isna().mean() * 100}).round(2)
    show(miss[(miss > 0).any(axis=1)])
    dups = {
        f"{PM_LABEL}: full rows": int(P.duplicated().sum()),
        f"{PM_LABEL}: date + station": int(P.duplicated(["date", "sensor_name"]).sum()),
        f"{PM_LABEL}: date + lat + lon": int(P.duplicated(["date", "latitude", "longitude"]).sum()),
        f"{PM_LABEL}: date + lat + lon + station": int(P.duplicated(["date", "latitude", "longitude", "sensor_name"]).sum()),
        f"{GEE_LABEL}: full rows": int(G.duplicated().sum()),
        f"{GEE_LABEL}: date + id": int(G.duplicated(["date", "id"]).sum()),
        f"{GEE_LABEL}: date + lat + lon": int(G.duplicated(["date", "lat", "lon"]).sum())}
    show(pd.Series(dups, name="duplicates").to_frame())
    for k, v in dups.items():
        log(rows, "audit_duplicates_raw", k, "count", v)
    for c, v in {**P.isna().mean() * 100, **G.isna().mean() * 100}.items():
        if v > 0:
            log(rows, "audit_missing_raw_pct", c, "missing_pct", round(v, 3))


def audit_distributions(P, G):
    """1.4 Target and predictor distributions (raw)."""
    print(P[TARGET].describe().round(2))
    print("\nPM2.5 <= 0   :", int((P[TARGET] <= PM25_MIN).sum()), "->", P.loc[P[TARGET] <= PM25_MIN, TARGET].value_counts().head(3).to_dict())
    print("PM2.5 >= 500 :", int((P[TARGET] >= PM25_MAX).sum()), "->", P.loc[P[TARGET] >= PM25_MAX, TARGET].value_counts().head(3).to_dict())
    show(G[["temperature_celsius", "pressure_mb", "wind_u", "wind_v", "NO2", "CO", "O3", "AOD"]].describe().T.round(5))


def audit_independence(P, rows):
    """1.5 Share of variance between stations, and day-to-day correlation within a station."""
    v = P[(P[TARGET] > PM25_MIN) & (P[TARGET] < PM25_MAX)].sort_values(["sensor_name", "date"]).copy()
    for label, col in (("raw", v[TARGET]), ("log1p", np.log1p(v[TARGET]))):
        gm = col.mean(); gsz = col.groupby(v.sensor_name).size(); gmean = col.groupby(v.sensor_name).mean()
        print(f"share of {label} PM2.5 variance explained by station: {((gsz * (gmean - gm) ** 2).sum() / ((col - gm) ** 2).sum()) * 100:.1f}%")
    v["prev"] = v.groupby("sensor_name")[TARGET].shift(1)
    v["gap_days"] = v.groupby("sensor_name").date.diff().dt.days
    r1 = v.loc[v.gap_days == 1, [TARGET, "prev"]].corr(method="spearman").iloc[0, 1]
    print(f"Spearman correlation between a day and the previous day, same station: {r1:.2f}")
    print("Observations per station:", v.groupby("sensor_name").size().describe().round(0).to_dict())
    log(rows, "audit_independence", "pm25", "lag1_spearman_same_station", round(r1, 3))


def audit_target_leakage(G):
    """1.6 No input derived from the target."""
    print("'pm25' column in the GEE file?", TARGET in G.columns)
    print("Inputs available:", [c for c in G.columns if c not in ("date", "datetime_utc", "id", "lat", "lon")])
    print("Features built later from the target: only pm25_lag1 and pm25_roll7_mean (strictly PAST days, see features.py).")


def make_station_ids(d, tol=STATION_TOL):
    """Within each station name, merge coordinates closer than `tol` degrees and split the rest."""
    coords = d[["sensor_name", "latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
    labels = np.ones(len(coords), dtype=int)
    for _, idx in coords.groupby("sensor_name").groups.items():
        if len(idx) > 1:
            labels[idx] = fcluster(linkage(coords.loc[idx, ["latitude", "longitude"]].values, "single"), t=tol, criterion="distance")
    coords["cluster"] = labels
    coords["n_clusters"] = coords.groupby("sensor_name").cluster.transform("nunique")
    coords["station_id"] = np.where(coords.n_clusters > 1, coords.sensor_name + " #" + coords.cluster.astype(str), coords.sensor_name)
    return d.merge(coords[["sensor_name", "latitude", "longitude", "station_id"]], on=["sensor_name", "latitude", "longitude"], how="left")


def audit_timestamps_and_ids(P, G, rows):
    """1.7 Timestamps, station identifiers and coordinates. Returns P with a reliable station_id, and the
    decision gate: is spatial CV allowed?"""
    print("unparseable dates:", P.date.isna().sum(), "(pm25)", G.date.isna().sum(), "(GEE)")
    print("GEE datetime_utc equals date everywhere:", bool((G.datetime_utc.dt.normalize() == G.date).all()))
    print("Coordinates plausible for the Mediterranean (lat 30-46, lon -10-20):",
          bool(P.latitude.between(30, 46).all() and P.longitude.between(-10, 20).all() and G.lat.between(30, 46).all() and G.lon.between(-10, 20).all()))
    print("GEE ids with more than one coordinate:", int((G.groupby("id")[["lat", "lon"]].nunique() > 1).any(axis=1).sum()), "of", G.id.nunique())

    chk = P.groupby("sensor_name")[["latitude", "longitude"]].nunique()
    multi = chk[(chk.latitude > 1) | (chk.longitude > 1)].index
    print("\nstation names with more than one coordinate:", len(multi), "of", P.sensor_name.nunique())
    sp = P[P.sensor_name.isin(multi)].groupby("sensor_name").agg(n_coords=("latitude", "nunique"), rows=(TARGET, "size"))
    for name in multi:
        c = P.loc[P.sensor_name == name, ["latitude", "longitude"]].drop_duplicates().values
        sp.loc[name, "max_spread_deg"] = max(np.hypot(*(a - b)) for a in c for b in c)
    show(sp.round(4).sort_values("max_spread_deg", ascending=False))

    P = make_station_ids(P)
    sc = P.groupby("station_id").agg(lat_spread=("latitude", lambda s: s.max() - s.min()), lon_spread=("longitude", lambda s: s.max() - s.min()))
    print("station names:", P.sensor_name.nunique(), "-> station_ids:", P.station_id.nunique())
    print("largest coordinate spread inside one station_id (deg):", round(max(sc.lat_spread.max(), sc.lon_spread.max()), 4))
    print(sorted(P.loc[P.sensor_name == "Barcelona", "station_id"].unique()))
    spatial_cv_ok = bool(max(sc.lat_spread.max(), sc.lon_spread.max()) <= STATION_TOL)
    print("\nDecision gate (section 3.1): reliable station IDs and coordinates ->", "spatial CV is allowed" if spatial_cv_ok else "OMIT spatial analyses and explain why")
    log(rows, "audit_ids", "station_names", "n", int(P.sensor_name.nunique()))
    log(rows, "audit_ids", "station_ids_after_disambiguation", "n", int(P.station_id.nunique()))
    log(rows, "audit_ids", "names_with_multiple_coordinates", "n", len(multi))
    return P, spatial_cv_ok


# ---------------------------------------------------------------- cleaning (section 5.1)
def remove_invalid(P, rows):
    """Remove missing / sentinel PM2.5 (<= 0 or >= 500) before EDA and splitting, then merge repeated
    readings of one station on one date by averaging."""
    clean = {"pm25_rows_raw": len(P),
             "removed_pm25_missing": int(P[TARGET].isna().sum()),
             "removed_pm25_le_0": int((P[TARGET] <= PM25_MIN).sum()),
             "removed_pm25_ge_500": int((P[TARGET] >= PM25_MAX).sum())}
    P = P[P[TARGET].notna() & (P[TARGET] > PM25_MIN) & (P[TARGET] < PM25_MAX)].copy()
    clean["pm25_rows_after_invalid_removal"] = len(P)
    show(pd.Series(clean, name="count").to_frame())

    dup_after = {"full rows": int(P.duplicated().sum()),
                 "date + station_id": int(P.duplicated(["date", "station_id"]).sum()),
                 "date + lat + lon": int(P.duplicated(["date", "latitude", "longitude"]).sum()),
                 "date + lat + lon + station_id": int(P.duplicated(["date", "latitude", "longitude", "station_id"]).sum())}
    show(pd.Series(dup_after, name="duplicates").to_frame())
    d = P[P.duplicated(["date", "station_id"], keep=False)].sort_values(["station_id", "date"])
    show(d[["date", "station_id", "latitude", "longitude", TARGET]].head(6))

    P = (P.groupby(["date", "station_id"], as_index=False)
           .agg(sensor_name=("sensor_name", "first"), latitude=("latitude", "median"), longitude=("longitude", "median"), pm25=(TARGET, "mean")))
    clean["duplicate_date_station_pairs_merged"] = dup_after["date + station_id"]
    clean["pm25_rows_unique_station_day"] = len(P)
    for k, v in clean.items():
        log(rows, "cleaning_counts", k, "count", v)
    print("unique station-days:", len(P), "| stations:", P.station_id.nunique())
    return P


# ---------------------------------------------------------------- matching (section 5.3.4)
def match_gee(P, G, rows):
    """One GEE ID per station (nearest grid point within MAX_DIST degrees), then join on date + GEE ID.
    Stations too far from any point, and station-days without a GEE record, are dropped (and counted)."""
    gee_pts = G[["id", "lat", "lon"]].drop_duplicates("id").reset_index(drop=True)
    st = P.groupby("station_id").agg(sensor_name=("sensor_name", "first"), lat=("latitude", "median"),
                                     lon=("longitude", "median"), n_readings=(TARGET, "size")).reset_index()
    dist = np.hypot(st.lat.values[:, None] - gee_pts.lat.values[None, :], st.lon.values[:, None] - gee_pts.lon.values[None, :])
    st["gee_id"] = gee_pts.id.values[dist.argmin(1)]
    st["dist_deg"] = dist.min(1)
    st["matched"] = st.dist_deg <= MAX_DIST
    print(f"stations matched within {MAX_DIST} deg: {st.matched.sum()} of {len(st)}")
    print("exact coordinate matches (distance 0):", int((st.dist_deg < 1e-6).sum()))
    print("\nunmatched stations (nearest GEE point too far):")
    show(st.loc[~st.matched, ["station_id", "lat", "lon", "dist_deg", "n_readings"]].round(3))

    assert st.station_id.is_unique                                   # one GEE ID per station: never mixed
    P2 = P.merge(st[["station_id", "gee_id", "dist_deg", "matched"]], on="station_id")
    n_unmatched_station = int((~P2.matched).sum())
    P2 = P2[P2.matched].drop(columns="matched")
    gcols = ["date", "id", "temperature_celsius", "pressure_mb", "wind_u", "wind_v", "NO2", "CO", "O3", "AOD"]
    df = P2.merge(G[gcols].rename(columns={"id": "gee_id"}), on=["date", "gee_id"], how="inner")
    df = df.sort_values(["station_id", "date"]).reset_index(drop=True)
    assert not df.duplicated(["date", "station_id"]).any()

    match_log = {"station_days_before_matching": len(P), "lost_station_too_far_from_any_GEE_point": n_unmatched_station,
                 "lost_no_GEE_record_for_that_date": len(P2) - len(df), "station_days_matched": len(df),
                 "stations_matched": df.station_id.nunique(), "gee_ids_used": df.gee_id.nunique()}
    for k, v in match_log.items():
        log(rows, "matching", k, "count", v)
    show(pd.Series(match_log, name="count").to_frame())

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(gee_pts.lon, gee_pts.lat, s=40, marker="s", c="lightgrey", label="GEE grid points")
    ax.scatter(st[st.matched].lon, st[st.matched].lat, s=12, c="tab:blue", label="matched stations")
    ax.scatter(st[~st.matched].lon, st[~st.matched].lat, s=30, c="tab:red", marker="x", label="unmatched stations")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.legend(); ax.set_title("Stations and GEE points")
    savefig("matching_map")
    return df


# ---------------------------------------------------------------- station timeline and retention (section 5.3.4)
def station_regularity(d):
    """Per station: first/last day, observed days, missing days, coverage and gaps."""
    out = []
    for s_, g in d.groupby("station_id"):
        dates = g.date.sort_values(); gaps = dates.diff().dt.days.dropna(); span = (dates.max() - dates.min()).days + 1
        out.append({"station_id": s_, "first": dates.min().date(), "last": dates.max().date(), "n_days": len(dates), "span_days": span,
                    "missing_days": span - len(dates), "coverage": len(dates) / span,
                    "median_gap_days": gaps.median() if len(gaps) else np.nan, "max_gap_days": gaps.max() if len(gaps) else np.nan})
    return pd.DataFrame(out).set_index("station_id")


def select_stations(df, regularity, rows):
    """Retention rule decided before splitting, from coverage only (never from PM2.5 values or test labels)."""
    alt = {}
    for n_, c_ in [(365, 0.5), (MIN_OBS, MIN_COVERAGE), (100, 0.0)]:
        k_ = regularity[(regularity.n_days >= n_) & (regularity.coverage >= c_)]
        alt[f"n>={n_}, coverage>={c_}"] = {"stations": len(k_), "rows": int(k_.n_days.sum())}
    show(pd.DataFrame(alt).T)
    keep = regularity[(regularity.n_days >= MIN_OBS) & (regularity.coverage >= MIN_COVERAGE)].index
    print(f"RULE: keep stations with >= {MIN_OBS} days and >= {MIN_COVERAGE*100:.0f}% coverage -> {len(keep)} of {len(regularity)} stations")
    df = df[df.station_id.isin(keep)].reset_index(drop=True)
    print("rows kept:", len(df))
    log(rows, "station_retention", "rule", "min_obs", MIN_OBS); log(rows, "station_retention", "rule", "min_coverage", MIN_COVERAGE)
    log(rows, "station_retention", "kept", "stations", len(keep)); log(rows, "station_retention", "kept", "rows", len(df))
    return df, keep


# ---------------------------------------------------------------- missing values (section 5.3.1)
def missingness_report(dev, features, rows):
    """Missingness by variable, and NO2/CO/O3 missingness by month, year and station (development data)."""
    mv = dev[features].isna().mean().mul(100).round(2).sort_values(ascending=False)
    show(mv[mv > 0].to_frame("missing %").T)
    gm = dev.groupby(dev.date.dt.month)[GAS].apply(lambda g: g.isna().mean().mul(100)).round(1)
    gy = dev.groupby(dev.date.dt.year)[GAS].apply(lambda g: g.isna().mean().mul(100)).round(1)
    gs = dev.groupby("station_id")[GAS].apply(lambda g: g.isna().mean().mul(100)).round(1)
    print("NO2/CO/O3 missing % by year:"); show(gy.T)
    print("NO2/CO/O3 missing % by month:"); show(gm.T)
    print("stations with most NO2 missing:"); show(gs.sort_values("NO2", ascending=False).head(5).T)
    for var, val in mv.items():
        log(rows, "missingness_dev_by_variable", var, "missing_pct", float(val))
    for nm, t_ in (("month", gm), ("year", gy)):
        for idx, r in t_.iterrows():
            for c_, val in r.items():
                log(rows, f"missingness_gas_by_{nm}", f"{c_}_{idx}", "missing_pct", float(val))
    for idx, r in gs.iterrows():
        for c_, val in r.items():
            log(rows, "missingness_gas_by_station", f"{c_}_{idx}", "missing_pct", float(val))
    return gm, gy, gs


def knn_imputer_check(dev, folds_temporal, rows):
    """KNNImputer is allowed only if the correlation structure of NO2/CO/O3 is stable across the training folds."""
    cols = GAS + ["AOD", "temperature_celsius", "pressure_mb", "wind_u", "wind_v"]
    ref = dev.loc[folds_temporal[-1][0], cols].corr(method="spearman")
    out = []
    for k, (tr, _) in enumerate(folds_temporal):
        t = dev.loc[tr, cols]; obs = float(t[GAS].notna().mean().min())
        c = t.corr(method="spearman") if obs > 0.05 else None
        diff = np.nan if c is None else float((c - ref).loc[GAS].abs().max().max())
        out.append({"training fold": k, "train_rows": len(tr), "min share of gas observed": round(obs, 2),
                    "max |corr - last fold| (gas rows)": round(diff, 3) if diff == diff else np.nan})
    stab = pd.DataFrame(out).set_index("training fold"); show(stab)
    knn_stable = bool((stab["min share of gas observed"] >= 0.5).all() and (stab.iloc[:, -1] <= 0.10).all())
    print("\nKNNImputer is", "JUSTIFIED" if knn_stable else "NOT justified: gas data are (almost) absent from the early training folds and correlations are not stable")
    print("=> use training-fold MEDIAN imputation + missingness indicators (the default for Sentinel-5P variables)")
    log(rows, "knn_imputer_check", "stability", "knn_justified", knn_stable)
    return knn_stable


def imputation_check(X_tr, X_va, features):
    """Median imputation + missingness indicators, learned on the training fold and applied to validation."""
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True).fit(X_tr)
    Xtr_i, Xva_i = imputer.transform(X_tr), imputer.transform(X_va)
    print("train", X_tr.shape, "validation", X_va.shape, "->", Xtr_i.shape[1], "columns after imputation (", Xtr_i.shape[1] - len(features), "missingness indicators )")
    print("missing values left in validation:", int(np.isnan(Xva_i).sum()))
    print("medians learned on the training fold (first 5):", dict(zip(features[:5], np.round(imputer.statistics_[:5], 4))))
    assert np.allclose(imputer.statistics_, X_tr.median().fillna(0).values, equal_nan=True)     # TRAINING medians, not full-data ones


# ---------------------------------------------------------------- scaling and feature selection (section 5.3.2)
def scaling_and_selection(X_tr, y_tr, features, topk=10):
    """Compare scalers, then run five selection methods (filter, wrapper, embedded) on the training fold."""
    imp0 = SimpleImputer(strategy="median", keep_empty_features=True).fit(X_tr)
    Xi = pd.DataFrame(imp0.transform(X_tr), columns=features)
    std = StandardScaler().fit(Xi); RobustScaler().fit(Xi)                    # statistics from TRAINING data
    cmp_ = pd.DataFrame({"mean": Xi.mean(), "std": Xi.std(), "median": Xi.median(), "IQR": Xi.quantile(.75) - Xi.quantile(.25)}).loc[["NO2", "CO", "AOD", "pressure_mb", "wind_u"]]
    show(cmp_.apply(lambda c: c.map(lambda v: float(f"{v:.4g}"))))
    print("NO2 spans ~1e-4 while pressure spans ~1e3: distance/gradient models would be dominated by pressure without scaling.")
    print("Choice: StandardScaler for SVR/KNN/neural nets/PCA/linear models. RobustScaler (median and IQR) is the alternative if outliers distort the mean; tree models: no scaling.")
    Xs = pd.DataFrame(std.transform(Xi), columns=features)

    # a random 8000-row sample of the training fold keeps MI/RFE/trees quick
    rng = np.random.RandomState(SEED); sub = rng.choice(len(Xs), min(8000, len(Xs)), replace=False)
    Xsub, ysub = Xs.iloc[sub], y_tr.iloc[sub]
    sel = pd.DataFrame(index=features)
    F, _ = f_regression(Xsub, ysub); sel["anova_F"] = pd.Series(F, index=features)
    sel["mutual_info"] = pd.Series(mutual_info_regression(Xsub, ysub, random_state=SEED), index=features)
    sel["rfe_rank"] = RFE(Ridge(alpha=1.0), n_features_to_select=topk).fit(Xsub, ysub).ranking_
    sel["lasso_coef"] = Lasso(alpha=0.1, max_iter=5000).fit(Xsub, ysub).coef_
    sel["tree_importance"] = ExtraTreesRegressor(100, max_depth=12, random_state=SEED, n_jobs=-1).fit(Xsub, ysub).feature_importances_

    chosen = {"anova_F": set(sel.anova_F.nlargest(topk).index), "mutual_info": set(sel.mutual_info.nlargest(topk).index),
              "rfe": set(sel.index[sel.rfe_rank == 1]), "lasso_L1": set(sel.index[sel.lasso_coef != 0]),
              "tree_importance": set(sel.tree_importance.nlargest(topk).index)}
    for m_, s_ in chosen.items():
        sel[f"sel_{m_}"] = sel.index.isin(s_)
    sel["votes"] = sel[[c for c in sel.columns if c.startswith("sel_")]].sum(axis=1)
    show(sel.sort_values("votes", ascending=False).head(15).round(4))
    rec = pd.DataFrame({"method": list(chosen), "n_selected": [len(v) for v in chosen.values()],
                        "selected_features": [", ".join(sorted(v)) for v in chosen.values()]})
    show(rec)
    print("Features chosen by at least 3 of 5 methods:", sorted(sel.index[sel.votes >= 3]))
    return sel, rec


# ---------------------------------------------------------------- the reusable pipeline
def make_preprocessor(features, scale=True, k="all"):
    """Median imputation + missingness indicators (+ StandardScaler) (+ SelectKBest). Fitted on a training fold only."""
    steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    parts = [("pre", ColumnTransformer([("num", Pipeline(steps), features)]))]
    if k != "all":
        parts.append(("select", SelectKBest(f_regression, k=k)))
    return Pipeline(parts)


# ---------------------------------------------------------------- outliers (section 5.3.3)
def outlier_treatment(dev, folds_temporal, features, y_tr, y_va, rows):
    """Thresholds learned on the training fold; compare MAE/RMSE/R2 with and without capping the training
    target at its 99.5th percentile, always evaluated on the untouched validation target."""
    q1, q3 = y_tr.quantile([.25, .75]); fence = q3 + 1.5 * (q3 - q1); cap = y_tr.quantile(.995)
    print(f"thresholds learned on the TRAINING fold: IQR fence {fence:.1f}, 99.5th percentile {cap:.1f} ug/m3")
    print(f"share above the fence: training {(y_tr > fence).mean()*100:.2f}% | validation {(y_va > fence).mean()*100:.2f}%  (applied unchanged)")

    def run(model_fn, scale, treat):
        out = []
        for tr, va in folds_temporal:
            Xt, Xv, yt, yv = dev.loc[tr, features], dev.loc[va, features], dev.loc[tr, TARGET], dev.loc[va, TARGET]
            if treat:
                yt = yt.clip(upper=yt.quantile(.995))                      # threshold from this training fold only
            pipe = Pipeline(make_preprocessor(features, scale).steps + [("model", model_fn())]).fit(Xt, yt)
            p = pipe.predict(Xv)
            out.append((mean_absolute_error(yv, p), np.sqrt(mean_squared_error(yv, p)), r2_score(yv, p)))
        return np.mean(out, axis=0)

    CAP = "cap training target at p99.5"
    res = {}
    for name, fn, sc_ in (("ElasticNet", lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=3000), True),
                          ("HistGradientBoosting", lambda: HistGradientBoostingRegressor(max_iter=100, random_state=SEED), False)):
        for treat in (False, True):
            res[(name, CAP if treat else "no treatment")] = run(fn, sc_, treat)
    outl_cmp = pd.DataFrame(res, index=["MAE", "RMSE", "R2"]).T.round(3)
    show(outl_cmp)
    for (m_, t_), r in outl_cmp.iterrows():
        for k_, v_ in r.items():
            log(rows, "outlier_treatment_comparison", f"{m_} | {t_}", k_, float(v_))

    gain = {m_: 100 * (1 - outl_cmp.loc[(m_, CAP), "RMSE"] / outl_cmp.loc[(m_, "no treatment"), "RMSE"]) for m_ in ("ElasticNet", "HistGradientBoosting")}
    print("held-out RMSE change from capping the training target (%, positive = better):", {k: round(float(v), 2) for k, v in gain.items()})
    adopt_cap = all(v > 2 for v in gain.values())
    print("Decision:", "adopt the cap" if adopt_cap else "keep the raw PM2.5 values. The effect is small and not consistent across models, and extreme days are real episodes that the model must learn.")
    log(rows, "outlier_treatment_decision", "cap_target_p99.5", "adopted", bool(adopt_cap))
    return outl_cmp


def pipeline_check(X_tr, y_tr, X_va, y_va, features):
    """Scale-sensitive model (scaled) vs tree model (not scaled) through the same Pipeline; checks no leakage."""
    pipe_scaled = Pipeline(make_preprocessor(features, scale=True).steps + [("model", Ridge(alpha=1.0))])
    pipe_trees = Pipeline(make_preprocessor(features, scale=False).steps + [("model", ExtraTreesRegressor(100, random_state=SEED, n_jobs=-1))])
    pipe_scaled.fit(X_tr, y_tr); pipe_trees.fit(X_tr, y_tr)
    for nm, p_ in (("Ridge + scaling", pipe_scaled), ("ExtraTrees, no scaling", pipe_trees)):
        pr = p_.predict(X_va)
        print(f"{nm:24s} RMSE {np.sqrt(mean_squared_error(y_va, pr)):.2f} | MAE {mean_absolute_error(y_va, pr):.2f} | R2 {r2_score(y_va, pr):.2f}   (last fold, untuned: only proves the pipeline runs)")
    fitted_imp = pipe_scaled.named_steps["pre"].named_transformers_["num"].named_steps["imputer"]
    assert np.allclose(fitted_imp.statistics_, X_tr.median().fillna(0).values, equal_nan=True)
    print("\nCheck: the pipeline's medians equal the TRAINING-fold medians -> no leakage from validation or test")
