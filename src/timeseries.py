"""Time-series exploration (section 5.2): timestamp regularity, ACF/PACF, Ljung-Box, ADF, drift, patterns,
lead-lag relations and the decision gate. Results go to results/Exploratory_temporal.csv."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, adfuller, pacf

from src.config import GAS, TARGET
from src.preprocessing import station_regularity
from src.utils import log, savefig, show

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def panels(df, P):
    """Per-station series indexed by date: target series before the GEE match (P), and the modelling table
    (df) on a regular daily calendar (gaps become NaN)."""
    print("timestamps are whole days (no time of day):", bool((df.date == df.date.dt.normalize()).all()))
    print("duplicate timestamps per station:", int(df.duplicated(["date", "station_id"]).sum()))
    daily = {s: g.set_index("date").sort_index().asfreq("D") for s, g in df.groupby("station_id")}
    panelP = {s: g.set_index("date").sort_index() for s, g in P.groupby("station_id")}
    dailyP = {s: g.asfreq("D") for s, g in panelP.items()}
    return daily, panelP, dailyP


def regularity(P, G, df, rows):
    """5a. Timestamp regularity, daily sampling, missing dates, irregular observations per station."""
    regularity_target = station_regularity(P)        # all valid PM2.5 readings (before matching)
    reg = station_regularity(df)                     # the modelling table (after matching to GEE)
    cols_ = ["n_days", "coverage", "missing_days", "median_gap_days", "max_gap_days"]
    show(pd.DataFrame({"target series (before GEE match)": regularity_target[cols_].median(),
                       "modelling table (after GEE match)": reg[cols_].median()}).round(2))
    span_days = pd.date_range(P.date.min(), P.date.max())
    gee_share = pd.Series(span_days).isin(set(G.date)).mean()
    print(f"GEE has a record for only {gee_share*100:.0f}% of the calendar days between {P.date.min().date()} and {P.date.max().date()}")

    all_days = pd.date_range(df.date.min(), df.date.max())
    per_day = df.groupby("date").size().reindex(all_days, fill_value=0)
    print(f"modelling-table calendar: {len(all_days)} days, {int((per_day == 0).sum())} days with no station at all")
    fig, ax = plt.subplots(1, 2, figsize=(14, 3.8))
    per_day.plot(ax=ax[0], title="stations reporting per day (modelling table)")
    ax[1].hist([regularity_target.coverage, reg.coverage], bins=20, label=["before GEE match", "after GEE match"]); ax[1].legend(); ax[1].set_title("coverage per station")
    savefig("ts_regularity")
    for nm, rg in (("target_series", regularity_target), ("modelling_table", reg)):
        for k, v_ in {"stations": len(rg), "median_coverage": rg.coverage.median(), "median_missing_days": rg.missing_days.median(),
                      "median_max_gap_days": rg.max_gap_days.median()}.items():
            log(rows, "timestamp_quality", nm, k, float(v_))
    log(rows, "timestamp_quality", "GEE", "share_of_calendar_days_with_record", float(gee_share))
    log(rows, "timestamp_quality", "modelling_table", "days_with_no_station", int((per_day == 0).sum()))
    for s_, r in reg.iterrows():
        for k in ("n_days", "coverage", "max_gap_days"):
            log(rows, "timestamp_quality_station", s_, k, float(r[k]))
    return reg


def longest_run(dates):
    """First day, last day and length of the longest run of consecutive days."""
    d = pd.Series(sorted(dates)); grp = (d.diff().dt.days != 1).cumsum()
    runs = d.groupby(grp).agg(["min", "max", "size"]); r = runs.loc[runs["size"].idxmax()]
    return r["min"], r["max"], int(r["size"])


def autocorrelation(P, panelP, dailyP, rows):
    """5b. ACF per station without crossing gaps; ADF and Ljung-Box on each station's longest unbroken run."""
    elig = [s_ for s_, g in P.groupby("station_id") if len(g) >= 100]
    acf_tab = pd.DataFrame({s_: [dailyP[s_][TARGET].corr(dailyP[s_][TARGET].shift(k)) for k in range(1, 15)] for s_ in elig}, index=range(1, 15)).T
    acf_sum = acf_tab.describe().T[["25%", "50%", "75%"]]
    show(acf_sum.head(7).round(3).T)
    ax = acf_sum["50%"].plot(marker="o", title=f"Median station ACF (band = quartiles), {len(elig)} stations")
    ax.fill_between(acf_sum.index, acf_sum["25%"], acf_sum["75%"], alpha=.25); ax.axhline(0, c="grey", lw=.5)
    savefig("ts_acf_stations")
    for k, r in acf_sum.iterrows():
        log(rows, "acf_median_over_stations", f"lag{k}", "median", float(r["50%"]))

    tests = []
    for s_ in elig:
        a, b, n = longest_run(panelP[s_].index)
        if n >= 60:
            x = panelP[s_].loc[a:b, TARGET]
            tests.append({"station_id": s_, "run_days": n, "adf_p": adfuller(x, autolag="AIC")[1],
                          "ljungbox_p_lag7": acorr_ljungbox(x, lags=[7]).lb_pvalue.iloc[0]})
    tests = pd.DataFrame(tests).set_index("station_id")
    print(f"{len(tests)} stations with a run of >= 60 consecutive days (median run {int(tests.run_days.median())} days)")
    print(f"ADF: stationary (p < 0.05) at {(tests.adf_p < .05).mean()*100:.0f}% of these stations")
    print(f"Ljung-Box lag 7: autocorrelation present (p < 0.05) at {(tests.ljungbox_p_lag7 < .05).mean()*100:.0f}% of these stations")
    for k, v_ in {"n_stations_tested": len(tests), "pct_adf_stationary": (tests.adf_p < .05).mean() * 100,
                  "pct_ljungbox_autocorrelated": (tests.ljungbox_p_lag7 < .05).mean() * 100, "median_run_days": tests.run_days.median()}.items():
        log(rows, "stationarity_tests", "per_station", k, float(v_))

    # classical ACF / PACF plots for the two stations with the longest unbroken runs
    fig, ax = plt.subplots(2, 2, figsize=(13, 6))
    for i, s_ in enumerate(tests.sort_values("run_days", ascending=False).head(2).index):
        a, b, n = longest_run(panelP[s_].index); x = panelP[s_].loc[a:b, TARGET]
        ax[i, 0].stem(range(21), acf(x, nlags=20)); ax[i, 0].set_title(f"ACF: {s_} ({n} consecutive days)")
        ax[i, 1].stem(range(21), pacf(x, nlags=20)); ax[i, 1].set_title("PACF")
        for a_ in ax[i]:
            a_.axhline(1.96 / np.sqrt(n), ls="--", c="grey"); a_.axhline(-1.96 / np.sqrt(n), ls="--", c="grey")
    savefig("ts_acf_pacf_examples")
    return acf_sum, tests


def _drift_stats(g):
    return pd.Series({"median": g.median(), "iqr": g.quantile(.75) - g.quantile(.25), "share_gt25_pct": (g > 25).mean() * 100,
                      "share_gt50_pct": (g > 50).mean() * 100, "n": len(g)})


def drift(P, rows):
    """5c. Target drift across years and months: median, IQR and high-concentration frequency."""
    by_year = P.groupby(P.date.dt.year)[TARGET].apply(_drift_stats).unstack()
    by_month = P.groupby(P.date.dt.month)[TARGET].apply(_drift_stats).unstack()
    show(by_year.round(2)); show(by_month.round(2).T)
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
    by_year[["median", "iqr"]].plot(marker="o", ax=ax[0], title="drift by year")
    by_month[["median", "iqr"]].plot(marker="o", ax=ax[1], title="drift by month")
    savefig("ts_drift")
    for per, t_ in (("year", by_year), ("month", by_month)):
        for idx, r in t_.iterrows():
            for m, val in r.items():
                log(rows, f"drift_by_{per}", idx, m, float(val))


def patterns(P, df, rows):
    """5d. Weekday, monthly and seasonal patterns of PM2.5 and the atmospheric predictors."""
    dow = P.assign(dow=P.date.dt.day_name()).groupby("dow")[TARGET].median().reindex(WEEKDAYS)
    show(dow.round(2).to_frame("median PM2.5 by weekday").T)
    pat_vars = [TARGET, "AOD"] + GAS + ["temperature_celsius", "pressure_mb", "wind_speed"]
    monthly = df.groupby("month")[pat_vars].median()
    z = (monthly - monthly.mean()) / monthly.std()
    fig, ax = plt.subplots(figsize=(10, 4)); im_ = ax.imshow(z.T, aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    ax.set_yticks(range(len(pat_vars))); ax.set_yticklabels(pat_vars); ax.set_xticks(range(12)); ax.set_xticklabels(range(1, 13)); ax.set_xlabel("month")
    plt.colorbar(im_, label="monthly median (z-score across months)"); ax.set_title("Monthly pattern of PM2.5 and predictors")
    savefig("ts_monthly_patterns")
    seasonal = df.groupby("season")[pat_vars].median().reindex(["winter", "spring", "summer", "autumn"])
    show(seasonal.round(5))
    for idx, r in seasonal.iterrows():
        for m, val in r.items():
            log(rows, "seasonal_median", idx, m, float(val))
    for idx, val in dow.items():
        log(rows, "weekday_median", idx, TARGET, float(val))


def lead_lag(df, daily, rows):
    """5e. Spearman(PM2.5 at t, predictor at t-k) per station, averaged. Positive k: the predictor comes first."""
    ll_vars = ["AOD", "NO2", "CO", "O3", "wind_speed", "temperature_celsius"]
    lags = range(-3, 4)
    elig = [s_ for s_, g in df.groupby("station_id") if len(g) >= 100]
    ll = pd.DataFrame(index=ll_vars, columns=list(lags), dtype=float)
    for x in ll_vars:
        for k in lags:
            r = []
            for s_ in elig:
                pair = pd.concat([daily[s_][TARGET], daily[s_][x].shift(k)], axis=1).dropna()
                if len(pair) >= 60:
                    r.append(pair.corr(method="spearman").iloc[0, 1])
            ll.loc[x, k] = np.nanmean(r) if r else np.nan
    show(ll.round(3))
    for x in ll_vars:
        for k in lags:
            log(rows, "lead_lag_spearman", x, f"lag{k}", float(ll.loc[x, k]))


def decision_gate(acf_sum, tests, reg, rows):
    """5f. Are temporal validation and lag-based features appropriate?"""
    acf1 = float(acf_sum.loc[1, "50%"])
    ac_share = float((tests.ljungbox_p_lag7 < .05).mean())
    cover = float(reg.coverage.median())
    temporal_ok = acf1 > 0.3 and ac_share > 0.5
    print(f"median lag-1 ACF across stations      : {acf1:.2f}  (rule: > 0.3)")
    print(f"stations with Ljung-Box p < 0.05      : {ac_share*100:.0f}%  (rule: > 50%)")
    print(f"median station coverage               : {cover*100:.0f}% of days between first and last reading")
    print("\n=> temporal validation and lag/rolling features are", "APPROPRIATE" if temporal_ok else "NOT justified")
    print("   Scenario: forecast PM2.5 for day t using information up to day t-1 (past PM2.5 allowed), with day-t")
    print("   satellite/meteorological inputs. For same-day NOWCASTING, use FEATURES_NOWCAST (no past PM2.5).")
    log(rows, "decision_gate", "temporal_validation_and_lags", "appropriate", bool(temporal_ok))
    log(rows, "decision_gate", "median_lag1_acf", "value", acf1)
    return temporal_ok
