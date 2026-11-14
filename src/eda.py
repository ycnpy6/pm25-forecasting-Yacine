"""Data exploration (section 5.1): summaries, plots, outliers, correlations, relationships with PM2.5,
target distribution. Results are logged into the rows that become results/Exploratory_Data.csv."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.config import KEY_VARS, SEED, TARGET
from src.utils import log, savefig, show

SEASONS = ["winter", "spring", "summer", "autumn"]
PLOT_VARS = KEY_VARS + ["wind_speed"]


def add_derived_columns(df):
    """wind speed (from predictors only) and calendar labels, used for plots and groupings."""
    df = df.copy()
    df["wind_speed"] = np.hypot(df.wind_u, df.wind_v)
    df["year"], df["month"] = df.date.dt.year, df.date.dt.month
    df["season"] = df.month.map({12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
                                 6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"})
    return df


def describe(df):
    """4.1 Describe the dataset."""
    print(f"rows {len(df)} | stations {df.station_id.nunique()} | GEE points {df.gee_id.nunique()} | "
          f"{df.date.min().date()} -> {df.date.max().date()}")
    print(f"lat {df.latitude.min():.2f}..{df.latitude.max():.2f} | lon {df.longitude.min():.2f}..{df.longitude.max():.2f}")
    print("Target: pm25 (ug/m3, continuous). Predictors:", KEY_VARS[1:])


def numerical_summary(df, rows):
    """4.2 min, max, mean, std, quartiles, skewness, kurtosis, missing %; counts of the categorical identifiers."""
    summ = df[KEY_VARS].describe().T
    summ["skewness"], summ["kurtosis"] = df[KEY_VARS].skew(), df[KEY_VARS].kurt()
    summ["missing_pct"] = df[KEY_VARS].isna().mean() * 100
    show(summ.round(5))
    for var, r in summ.iterrows():
        for m, val in r.items():
            log(rows, "numerical_summary", var, m, float(val))

    cat_counts = {"station": df.station_id.value_counts(), "year": df.year.value_counts().sort_index(),
                  "month": df.month.value_counts().sort_index(), "season": df.season.value_counts()}
    for k, s in cat_counts.items():
        for item, val in s.items():
            log(rows, f"category_counts_{k}", item, "rows", int(val))
    show(df.year.value_counts().sort_index().to_frame("rows").T)
    print("stations: median", int(cat_counts["station"].median()), "rows | min", cat_counts["station"].min(), "| max", cat_counts["station"].max())


def distribution_plots(df):
    """4.3 Histograms, boxplots, categorical plots, PM2.5 by station."""
    fig, axes = plt.subplots(2, 5, figsize=(17, 6))
    for ax, c in zip(axes.ravel(), PLOT_VARS):
        ax.hist(df[c].dropna(), bins=50); ax.set_title(c)
    savefig("eda_histograms")

    fig, axes = plt.subplots(2, 5, figsize=(17, 6))
    for ax, c in zip(axes.ravel(), PLOT_VARS):
        ax.boxplot(df[c].dropna(), vert=True); ax.set_title(c)
    savefig("eda_boxplots")

    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
    df.year.value_counts().sort_index().plot.bar(ax=ax[0], title="rows per year")
    df.month.value_counts().sort_index().plot.bar(ax=ax[1], title="rows per month")
    df.groupby("season")[TARGET].median().reindex(SEASONS).plot.bar(ax=ax[2], title="median PM2.5 by season")
    df.station_id.value_counts().plot.hist(bins=30, ax=ax[3], title="readings per station")
    savefig("eda_categorical_plots")

    top = df.station_id.value_counts().head(25).index
    df[df.station_id.isin(top)].boxplot(column=TARGET, by="station_id", rot=90, figsize=(14, 4), showfliers=False)
    plt.suptitle(""); plt.title("PM2.5 by station (25 stations with most readings, outliers hidden)")
    savefig("eda_pm25_by_station")


def outliers(df, rows):
    """4.4 Missingness and potential outliers (IQR, z-scores, Isolation Forest). Inspected, not removed."""
    mv = df[PLOT_VARS].isna().mean().mul(100).round(2)
    print("Missing % per variable:", mv[mv > 0].to_dict())
    out = []
    for c in PLOT_VARS:
        s = df[c].dropna(); q1, q3 = s.quantile([.25, .75]); i = q3 - q1
        z = (s - s.mean()) / s.std()
        out.append({"variable": c, "iqr_low": q1 - 1.5 * i, "iqr_high": q3 + 1.5 * i,
                    "n_iqr_outliers": int(((s < q1 - 1.5 * i) | (s > q3 + 1.5 * i)).sum()),
                    "pct_iqr_outliers": 100 * ((s < q1 - 1.5 * i) | (s > q3 + 1.5 * i)).mean(), "n_abs_z_gt_3": int((z.abs() > 3).sum())})
    outl = pd.DataFrame(out).set_index("variable")
    show(outl.round(4))
    for var, r in outl.iterrows():
        for m, val in r.items():
            log(rows, "outliers_iqr_z", var, m, float(val))

    # Isolation Forest on the predictors (median-filled copy, for INSPECTION only, never used for modelling)
    Xif = df[KEY_VARS[1:] + ["wind_speed"]].copy()
    Xif = Xif.fillna(Xif.median())
    iso = IsolationForest(contamination=0.01, random_state=SEED, n_jobs=-1).fit(StandardScaler().fit_transform(Xif))
    flag = iso.predict(StandardScaler().fit_transform(Xif)) == -1
    print(f"Isolation Forest flags {flag.sum()} rows ({flag.mean()*100:.1f}%)")
    print("median PM2.5 of flagged rows:", round(df.loc[flag, TARGET].median(), 2), "| of the others:", round(df.loc[~flag, TARGET].median(), 2))
    print("Highest PM2.5 values (candidate episodes or errors):")
    show(df.nlargest(8, TARGET)[["date", "station_id", TARGET, "AOD", "wind_speed"]])
    log(rows, "outliers_isolation_forest", "predictors", "flagged_rows", int(flag.sum()))


def correlations(df, rows):
    """4.5 Pearson, Spearman, pairs with |r| > 0.9, VIF, PCA."""
    pe, sp_ = df[KEY_VARS].corr(method="pearson"), df[KEY_VARS].corr(method="spearman")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    for a, m, t in zip(ax, (pe, sp_), ("Pearson", "Spearman")):
        a.imshow(m, cmap="coolwarm", vmin=-1, vmax=1)
        a.set_xticks(range(len(KEY_VARS))); a.set_xticklabels(KEY_VARS, rotation=90); a.set_yticks(range(len(KEY_VARS))); a.set_yticklabels(KEY_VARS)
        for i in range(len(KEY_VARS)):
            for j in range(len(KEY_VARS)):
                a.text(j, i, f"{m.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
        a.set_title(t)
    savefig("correlation_matrix")

    both = pd.DataFrame({"pearson": pe[TARGET], "spearman": sp_[TARGET]}).drop(TARGET).sort_values("spearman")
    show(both.round(3))
    pairs = sp_.where(np.triu(np.ones(sp_.shape), 1).astype(bool)).stack()
    pairs_p = pe.where(np.triu(np.ones(pe.shape), 1).astype(bool)).stack()
    hi = pd.concat([pairs[pairs.abs() > 0.9], pairs_p[pairs_p.abs() > 0.9]])
    print("Predictor pairs with |r| > 0.9 (Pearson or Spearman):", hi.to_dict() or "none")
    for var in both.index:
        log(rows, "correlation_with_pm25", var, "pearson", float(both.loc[var, "pearson"]))
        log(rows, "correlation_with_pm25", var, "spearman", float(both.loc[var, "spearman"]))

    # VIF and PCA on the predictors (median-filled and standardised copy, for inspection only)
    pred = KEY_VARS[1:]
    Xs = StandardScaler().fit_transform(df[pred].fillna(df[pred].median()))
    vif = pd.Series([variance_inflation_factor(Xs, i) for i in range(Xs.shape[1])], index=pred, name="VIF")
    show(vif.round(2).to_frame().T)
    evr = PCA().fit(Xs).explained_variance_ratio_
    print("PCA cumulative explained variance:", np.round(np.cumsum(evr), 2))
    for var, val in vif.items():
        log(rows, "vif", var, "vif", float(val))


def _sp_corr(g, a="AOD"):
    return g[TARGET].corr(g[a], method="spearman") if len(g) > 30 else np.nan


def pm25_vs_predictors(df, rows):
    """4.6 PM2.5 versus predictors; does the PM2.5-AOD link change with season, station and weather?"""
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].hexbin(df.AOD.clip(upper=1), df[TARGET].clip(upper=60), gridsize=40, bins="log", cmap="viridis")
    ax[0].set_xlabel("AOD (clipped at 1)"); ax[0].set_ylabel("PM2.5 (clipped at 60)"); ax[0].set_title("PM2.5 vs AOD")
    ax[1].hexbin(df.NO2.dropna().clip(upper=3e-4), df.loc[df.NO2.notna(), TARGET].clip(upper=60), gridsize=40, bins="log", cmap="viridis")
    ax[1].set_xlabel("NO2"); ax[1].set_title("PM2.5 vs NO2")
    savefig("eda_pm25_vs_predictors")

    season = df.groupby("season").apply(_sp_corr, include_groups=False).reindex(SEASONS)
    print("Spearman(PM2.5, AOD) by season:"); show(season.round(3).to_frame("rho").T)
    bys = df.groupby("station_id").apply(lambda g: _sp_corr(g) if len(g) >= 100 else np.nan, include_groups=False).dropna()
    print(f"Spearman(PM2.5, AOD) per station ({len(bys)} stations with >= 100 rows): median {bys.median():.2f}, "
          f"share positive {(bys > 0).mean()*100:.0f}%, range {bys.min():.2f} to {bys.max():.2f}")

    meteo = []                          # correlation inside terciles of each meteorological variable
    for m in ["wind_speed", "temperature_celsius", "pressure_mb"]:
        grp = pd.qcut(df[m], 3, labels=["low", "mid", "high"])
        r = df.groupby(grp, observed=True).apply(_sp_corr, include_groups=False)
        meteo.append(r.rename(m))
        for lvl, val in r.items():
            log(rows, "aod_pm25_corr_by_meteorology", f"{m}_{lvl}", "spearman", float(val))
    show(pd.DataFrame(meteo).round(3))
    for s_, v_ in season.items():
        log(rows, "aod_pm25_corr_by_season", s_, "spearman", float(v_))
    log(rows, "aod_pm25_corr_by_station", "median", "spearman", float(bys.median()))


def target_distribution(df, rows):
    """4.7 The target distribution: skewness, share of high days, log1p view, monthly profile."""
    pm = df[TARGET]
    sh = {"mean": pm.mean(), "median": pm.median(), "std": pm.std(), "skewness": pm.skew(), "kurtosis": pm.kurt(),
          "share_gt_25_pct": (pm > 25).mean() * 100, "share_gt_50_pct": (pm > 50).mean() * 100, "max": pm.max()}
    show(pd.Series(sh, name="PM2.5").round(3).to_frame().T)
    for k, v_ in sh.items():
        log(rows, "target_distribution", TARGET, k, float(v_))

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    ax[0].hist(pm, bins=100); ax[0].set_title("PM2.5 raw (strongly right-skewed)")
    ax[1].hist(pm[pm < 40], bins=60); ax[1].set_title("PM2.5 below 40")
    ax[2].hist(np.log1p(pm), bins=60); ax[2].set_title("log(1 + PM2.5)")
    savefig("eda_target_distribution")

    mon = df.groupby("month")[TARGET].agg(median="median", mean="mean", share_gt25=lambda s: (s > 25).mean() * 100)
    show(mon.round(2).T)
