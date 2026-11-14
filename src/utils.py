"""Small helpers: result logging, figure saving and table display (work in scripts and notebooks)."""
import warnings

import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)
plt.rcParams["figure.figsize"] = (9, 4)


def log(rows, section, item, metric, value):
    """One line of Exploratory_Data.csv / Exploratory_temporal.csv."""
    rows.append({"section": section, "item": item, "metric": metric, "value": value})


def save_log(rows, path, replace=True):
    """Write the logged rows. With replace=False, rows of other sections already in the file are kept."""
    new = pd.DataFrame(rows)
    if not replace and path.exists():
        old = pd.read_csv(path, dtype=str, keep_default_na=False)
        new = pd.concat([old[~old.section.isin(new.section.unique())], new])
    new.to_csv(path, index=False)


def savefig(name):
    FIGURES.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES / f"{name}.png", dpi=110)
    plt.show()                  # displayed in notebooks, no effect in scripts
    plt.close("all")


def show(obj):
    """display() in a notebook, print() in a script."""
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            from IPython.display import display
            display(obj)
            return
    except ImportError:
        pass
    print(obj.to_string() if hasattr(obj, "to_string") else obj)
    print()
