"""Shared feature engineering for the HDB resale price predictor.

Both model.py (training) and app.py (serving) import from this file, so the
model always receives features built in exactly the same way. Training and
serving code drifting apart is one of the most common bugs in deployed ML.
"""

import json
import urllib.request
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent / "data" / "hdb_resale_2017_2019.csv"

# Newer sales (after May 2019 up to the latest month), downloaded from
# data.gov.sg by running:  python hdb_features.py
RECENT_DATA_PATH = Path(__file__).parent / "data" / "hdb_resale_2019_onwards.csv"
DATAGOV_DATASET_ID = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"  # Resale prices, Jan 2017 onwards
DATAGOV_DOWNLOAD_URL = (
    f"https://api-open.data.gov.sg/v1/public/api/datasets/{DATAGOV_DATASET_ID}/poll-download"
)

# Original source of the CSV (a mirror of data.gov.sg, Jan 2017 - May 2019).
# A local copy is kept in data/ so the lesson still works if this link breaks.
DATA_URL = (
    "https://raw.githubusercontent.com/kohjiaxuan/"
    "Predicting-HDB-Price-with-Machine-Learning/master/"
    "resale-flat-prices-based-on-registration-date-from-jan-2017-onwards.csv"
)

CATEGORICAL_FEATURES = ["town", "flat_type"]
NUMERIC_FEATURES = ["floor_area_sqm", "storey", "remaining_lease", "months_since_2017"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "resale_price"
LEASE_YEARS = 99  # every HDB flat has a 99-year lease

# def load_data(path=DATA_PATH, recent_path=RECENT_DATA_PATH):
#     """Load the 2017-2019 resale CSV (local copy, falling back to the URL),
#     plus the newer data.gov.sg sales if they have been downloaded."""
#     df = pd.read_csv(path) if Path(path).exists() else pd.read_csv(DATA_URL)
#     if recent_path and Path(recent_path).exists():
#         df = pd.concat([df, pd.read_csv(recent_path)], ignore_index=True)
#     return df

def load_data(path=DATA_PATH, recent_path=RECENT_DATA_PATH, last_n_months=None):
    """Load the 2017-2019 resale CSV (local copy, falling back to the URL),
    plus the newer data.gov.sg sales if they have been downloaded.
    If last_n_months is given, keep only that many of the latest months."""
    df = pd.read_csv(path) if Path(path).exists() else pd.read_csv(DATA_URL)
    if recent_path and Path(recent_path).exists():
        df = pd.concat([df, pd.read_csv(recent_path)], ignore_index=True)
    if last_n_months:
        month = pd.to_datetime(df["month"])
        df = df[month > month.max() - pd.DateOffset(months=last_n_months)]
    return df.reset_index(drop=True)



def download_recent_data(path=DATA_PATH, out_path=RECENT_DATA_PATH):
    """Download every sale from data.gov.sg, keep only the months after the
    2017-2019 CSV ends, and save them to data/ so training works offline."""
    # data.gov.sg rejects Python's default User-Agent with 403 Forbidden
    request = urllib.request.Request(DATAGOV_DOWNLOAD_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        csv_url = json.load(response)["data"]["url"]
    latest = pd.read_csv(csv_url)

    last_month = pd.read_csv(path, usecols=["month"])["month"].max()
    recent = latest[latest["month"] > last_month]  # "YYYY-MM" strings sort by date
    recent.to_csv(out_path, index=False)
    print(
        f"Saved {len(recent):,} sales ({recent['month'].min()} to "
        f"{recent['month'].max()}) to {Path(out_path).relative_to(Path(__file__).parent)}"
    )
    return recent


def storey_midpoint(storey_range):
    """Turn a storey band such as '10 TO 12' into its midpoint (11.0)."""
    bounds = storey_range.str.split(" TO ", expand=True).astype(int)
    return (bounds[0] + bounds[1]) / 2


# def add_features(df):
#     """Return a copy of the raw data with the engineered 'storey' column."""
#     df = df.copy()
#     df["storey"] = storey_midpoint(df["storey_range"])
#     return df
def add_sale_date_features(df):
    """Add features that depend on when the flat was sold.

    Needs 'month' ('YYYY-MM') and 'lease_commence_date'. The app calls this
    too, so a prediction uses exactly the same formulas as training.
    """
    df = df.copy()
    sale = pd.to_datetime(df["month"])
    df["months_since_2017"] = (sale.dt.year - 2017) * 12 + (sale.dt.month - 1)
    sale_year = sale.dt.year + (sale.dt.month - 1) / 12
    # Only the lease year is known, so assume the lease started mid-year
    df["remaining_lease_years"] = df["lease_commence_date"] + 0.5 + LEASE_YEARS - sale_year
    return df




def add_features(df):
    """Return a copy of the raw data with the engineered columns."""
    df = df.copy()
    df["storey"] = storey_midpoint(df["storey_range"])
    # "61 years 04 months" -> 61.33
    parts = df["remaining_lease"].str.extract(r"(\d+)\s*years?(?:\s*(\d+)\s*months?)?")
    df["remaining_lease"] = parts[0].astype(int) + parts[1].fillna(0).astype(int) / 12
    sale = pd.to_datetime(df["month"])
    df["months_since_2017"] = (sale.dt.year - 2017) * 12 + (sale.dt.month - 1)
    return df


if __name__ == "__main__":
    download_recent_data()
