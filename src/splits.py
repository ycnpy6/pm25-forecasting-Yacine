"""Final test period and validation folds (section 5.4). Folds are generated once, saved as JSON in folds/
(row positions inside the development table) and reused by every model."""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit

from src.config import FOLDS, N_FOLDS, SEED, TEST_START
from src.utils import show

FOLD_FILES = {"temporal": "folds_temporal.json", "spatial": "folds_spatial.json", "standard": "folds_standard.json"}


def split_dev_test(data, test_start=TEST_START):
    """Everything from test_start onward is the final test period, never used during development."""
    dev = data[data.date < test_start].reset_index(drop=True)
    test = data[data.date >= test_start].reset_index(drop=True)
    print(f"development: {len(dev)} rows, {dev.station_id.nunique()} stations, {dev.date.min().date()} -> {dev.date.max().date()}")
    print(f"final test  : {len(test)} rows, {test.station_id.nunique()} stations, {test.date.min().date()} -> {test.date.max().date()}")
    return dev, test


def make_folds(dev, spatial_ok=True):
    """temporal: TimeSeriesSplit on unique dates with a 1-day gap (mandatory);
    spatial: GroupKFold by station_id (mandatory if station IDs are reliable);
    standard: shuffled KFold (diagnostic only)."""
    udates = np.sort(dev.date.unique())
    temporal = [(np.where(dev.date.isin(udates[tr]))[0].tolist(), np.where(dev.date.isin(udates[va]))[0].tolist())
                for tr, va in TimeSeriesSplit(n_splits=N_FOLDS, gap=1).split(udates)]
    spatial = ([(a.tolist(), b.tolist()) for a, b in GroupKFold(N_FOLDS).split(dev, groups=dev.station_id)] if spatial_ok else None)
    standard = [(a.tolist(), b.tolist()) for a, b in KFold(N_FOLDS, shuffle=True, random_state=SEED).split(dev)]
    return {"temporal": temporal, "spatial": spatial, "standard": standard}


def check_folds(dev, folds, test_start=TEST_START):
    for tr, va in folds["temporal"]:                                    # validation strictly after training (+ gap)
        assert dev.date.iloc[tr].max() < dev.date.iloc[va].min()
    assert dev.date.max() < pd.Timestamp(test_start)                    # test period absent from every fold
    if folds["spatial"]:
        for tr, va in folds["spatial"]:
            assert not set(dev.station_id.iloc[tr]) & set(dev.station_id.iloc[va])
    print("checks passed: temporal order, no test dates in folds, no station shared by spatial train/validation")


def describe_folds(dev, folds):
    show(pd.DataFrame([{"fold": k, "train_rows": len(tr), "train_to": dev.date.iloc[tr].max().date(), "val_rows": len(va),
                        "val_from": dev.date.iloc[va].min().date(), "val_to": dev.date.iloc[va].max().date(),
                        "val_stations": dev.station_id.iloc[va].nunique()} for k, (tr, va) in enumerate(folds["temporal"])]))
    if folds["spatial"]:
        show(pd.DataFrame([{"fold": k, "train_stations": dev.station_id.iloc[tr].nunique(),
                            "val_stations": dev.station_id.iloc[va].nunique(), "val_rows": len(va)}
                           for k, (tr, va) in enumerate(folds["spatial"])]))


def save_folds(folds):
    FOLDS.mkdir(exist_ok=True)
    for name, f in FOLD_FILES.items():
        with open(FOLDS / f, "w") as fh:
            json.dump(folds[name], fh)
    print("saved", ", ".join(FOLD_FILES.values()), "in", FOLDS.name + "/")


def load_folds():
    out = {}
    for name, f in FOLD_FILES.items():
        with open(FOLDS / f) as fh:
            out[name] = json.load(fh)
    return out
