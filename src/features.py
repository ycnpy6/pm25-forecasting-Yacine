"""Feature engineering (section 5.3.4): lags, differences, rolling statistics, rate of change, cyclical calendar,
time since a high-AOD day, and past PM2.5. Every feature uses PAST information only, built per station on a
regular daily calendar, so a lag across a missing day is NaN, never a value copied from earlier."""
import numpy as np
import pandas as pd

from src.config import GAS, HIGH_AOD, MET, TARGET

DERIVED = ["AOD_lag1", "AOD_lag2", "AOD_lag3", "NO2_lag1", "AOD_diff1", "temp_diff1", "NO2_diff1", "AOD_roll3_mean", "AOD_roll7_mean",
           "AOD_roll7_std", "AOD_roll7_max", "AOD_chg3", "days_since_high_aod"]
PAST_TARGET = ["pm25_lag1", "pm25_roll7_mean"]       # allowed only because we forecast at least one day ahead
FAMILIES = {"AOD": ["AOD"], "NO2_CO_O3": GAS, "meteorological": MET,      # = the ablation families (section 8.2)
            "calendar": ["doy_sin", "doy_cos"], "lag_rolling": DERIVED + PAST_TARGET}
FEATURES = [c for cols in FAMILIES.values() for c in cols]                # forecasting (>= 1 day ahead)
FEATURES_NOWCAST = [c for c in FEATURES if c not in PAST_TARGET]          # same-day nowcasting: no past PM2.5


def daily_target_series(P):
    """Full target series of each station (before the GEE match) on a regular daily calendar."""
    return {s: g.set_index("date").sort_index().asfreq("D") for s, g in P.groupby("station_id")}


def station_features(s_, g, dailyP):
    g = g.set_index("date").sort_index().asfreq("D")            # regular daily calendar; gap days are NaN rows
    o = pd.DataFrame(index=g.index)
    for k in (1, 2, 3):
        o[f"AOD_lag{k}"] = g["AOD"].shift(k)
    o["NO2_lag1"] = g["NO2"].shift(1)
    o["AOD_diff1"] = g["AOD"].shift(1) - g["AOD"].shift(2)
    o["temp_diff1"] = g["temperature_celsius"].shift(1) - g["temperature_celsius"].shift(2)
    o["NO2_diff1"] = g["NO2"].shift(1) - g["NO2"].shift(2)
    past = g["AOD"].shift(1)                                    # every window ends at t-1
    o["AOD_roll3_mean"] = past.rolling(3, min_periods=2).mean()
    o["AOD_roll7_mean"] = past.rolling(7, min_periods=3).mean()
    o["AOD_roll7_std"] = past.rolling(7, min_periods=3).std()
    o["AOD_roll7_max"] = past.rolling(7, min_periods=3).max()
    o["AOD_chg3"] = g["AOD"].shift(1) - g["AOD"].shift(4)
    day = g.index.to_series()
    last_high = day.where(g["AOD"] > HIGH_AOD).shift(1).ffill()  # last high-AOD day strictly before t
    o["days_since_high_aod"] = (day - last_high).dt.days.clip(upper=30)
    x = dailyP[s_][TARGET]                                      # past PM2.5 (t-1 and earlier) only
    hist = pd.DataFrame({"pm25_lag1": x.shift(1), "pm25_roll7_mean": x.shift(1).rolling(7, min_periods=3).mean()})
    out = g.join(o).join(hist)
    out["station_id"] = s_
    return out.reset_index()


def build_features(df, P):
    """Modelling table: one row per observed station-day, with all features."""
    print(f"share of days with AOD > {HIGH_AOD}: {(df.AOD > HIGH_AOD).mean()*100:.1f}%")
    dailyP = daily_target_series(P)
    data = pd.concat([station_features(s_, g, dailyP) for s_, g in df.groupby("station_id")], ignore_index=True)
    data = data.dropna(subset=[TARGET])                          # filler calendar days are not observations
    doy = data.date.dt.dayofyear
    data["doy_sin"], data["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    data = data.sort_values(["date", "station_id"]).reset_index(drop=True)
    print("modelling table:", data.shape)
    print(len(FEATURES), "features (forecast) |", len(FEATURES_NOWCAST), "features (nowcast)")
    for fam, cols in FAMILIES.items():
        print(f"  {fam:15s}", cols if len(cols) < 6 else f"{len(cols)} features")
    assert "sensor_name" not in FEATURES and "station_id" not in FEATURES and "latitude" not in FEATURES   # IDs/coordinates are not predictors
    return data


def check_no_leakage(data, P, stations, n_stations=15):
    """pm25_lag1 must equal the real reading of the previous calendar day, and be NaN when that day is missing."""
    bad_lag = bad_gap = 0
    lookup = P.set_index(["station_id", "date"])
    for s in list(stations)[:n_stations]:
        g = data[data.station_id == s]
        key = pd.MultiIndex.from_arrays([g.station_id, g.date - pd.Timedelta(days=1)])
        prev_pm = pd.Series(lookup[TARGET].reindex(key).values, index=g.index)
        bad_lag += int((g["pm25_lag1"].fillna(-1) != prev_pm.fillna(-1)).sum())
        bad_gap += int((prev_pm.isna() & g["pm25_lag1"].notna()).sum())
    print("lag1 mismatches:", bad_lag, "| lags taken across a gap:", bad_gap)
    assert bad_lag == 0 and bad_gap == 0
    print("pm25_lag1 uses only the previous calendar day. corr(pm25, pm25_lag1) =",
          round(data[TARGET].corr(data["pm25_lag1"], method="spearman"), 3), "(< 1: it is yesterday, not today)")
