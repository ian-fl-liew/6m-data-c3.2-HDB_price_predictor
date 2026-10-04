"""Streamlit web app for the HDB resale price model.

Run locally with:  streamlit run app.py
Train the model first with:  python model.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from hdb_features import FEATURES

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "hdb_price_model.joblib"
CARD_PATH = MODEL_DIR / "model_card.json"

st.set_page_config(page_title="HDB Resale Price Predictor", page_icon="🏡")


# Cache so the model loads once, not on every click.
# mtime is part of the cache key, so a retrained model is picked up automatically.
@st.cache_resource
def load_model(mtime):
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_card(mtime):
    return json.loads(CARD_PATH.read_text())


if not MODEL_PATH.exists() or not CARD_PATH.exists():
    st.error("No trained model found. Run `python model.py` first, then reload this page.")
    st.stop()

# model = load_model()
# card = load_card()
# ranges = card["numeric_ranges"]
model = load_model(MODEL_PATH.stat().st_mtime)
card = load_card(CARD_PATH.stat().st_mtime)

# A model trained with old features would crash or give wrong answers
if card["features"] != FEATURES:
    st.error("The saved model uses different features from the app. Run `python model.py` again.")
    st.stop()
ranges = card["numeric_ranges"]

best = card["best_model"]
best_results = card["results"][best]

st.title("🏡 Singapore HDB Resale Price Predictor")
st.write(
    f"Estimate an HDB flat's resale price using a **{best}** model trained on "
    f"{card['data']['rows']:,} resale transactions "
    f"({card['data']['first_month']} to {card['data']['last_month']})."
)

predict_tab, compare_tab = st.tabs(["Predict a price", "Model comparison"])

with predict_tab:
    with st.form("flat_details"):
        col1, col2 = st.columns(2)
        with col1:
            town = st.selectbox("Town", card["categories"]["town"])
            flat_type = st.selectbox(
                "Flat type",
                card["categories"]["flat_type"],
                index=card["categories"]["flat_type"].index("4 ROOM"),
            )
            # storey = st.slider("Storey (floor level)", min_value=1, max_value=51, value=8)
            storey = st.slider(
                "Storey (floor level)",
                min_value=int(ranges["storey"]["min"]),
                max_value=int(ranges["storey"]["max"]),
                value=int(ranges["storey"]["median"]),
            )


        with col2:
            floor_area = st.slider(
                "Floor area (sqm)",
                min_value=int(ranges["floor_area_sqm"]["min"]),
                max_value=int(ranges["floor_area_sqm"]["max"]),
                value=int(ranges["floor_area_sqm"]["median"]),
            )
            # lease_year = st.slider(
            #     "Lease commencement year",
            #     min_value=int(ranges["lease_commence_date"]["min"]),
            #     max_value=int(ranges["lease_commence_date"]["max"]),
            #     value=int(ranges["lease_commence_date"]["median"]),
            # )
            remaining_lease = st.slider(
                "Remaining lease (years)",
                min_value=int(ranges["remaining_lease"]["min"]),
                max_value=int(ranges["remaining_lease"]["max"]),
                value=int(ranges["remaining_lease"]["median"]),
            )

        submitted = st.form_submit_button("Predict resale price", type="primary")

    if submitted:
        # month_index = (int(card["data"]["last_month"][:4]) - 2017) * 12 + int(card["data"]["last_month"][5:]) - 1

        # # Column names and order must match the training data exactly.
        # flat = pd.DataFrame(
        #     [[town, flat_type, floor_area, storey, remaining_lease, month_index]], columns=FEATURES
        # )
        # Price the flat at the latest month the model has seen, e.g. "2026-10" -> 117
        last_month = card["data"]["last_month"]
        month_index = (int(last_month[:4]) - 2017) * 12 + int(last_month[5:]) - 1
        # Column names and order must match the training data exactly.
        flat = pd.DataFrame(
            [[town, flat_type, floor_area, storey, remaining_lease, month_index]], columns=FEATURES
        )

        price = model.predict(flat)[0]
        typical_error = best_results["test_mae"]

        st.success(f"Estimated resale price: **S${price:,.0f}**")
        st.caption(
            f"On unseen test data this model was off by about S${typical_error:,.0f} "
            f"on average (MAE), so a realistic range is roughly "
            f"S${price - typical_error:,.0f} to S${price + typical_error:,.0f}."
        )
        # st.info(
        #     f"Prices reflect the {card['data']['first_month']} to "
        #     f"{card['data']['last_month']} market, not today's prices."
        # )
        st.info(
            f"Estimated at {last_month} prices, the latest month in the "
            f"training data. Retrain monthly to keep this current."
        )

with compare_tab:
    st.write(
        # "Three models were trained on the same data. The deployed model was "
        f"{len(card['results'])} models were trained on the same data. The deployed model was "
        f"chosen by **{card['selection_rule']}**."
    )
    table = pd.DataFrame.from_dict(card["results"], orient="index").rename(
        columns={
            "cv_rmse": "CV RMSE (S$)",
            "cv_mae": "CV MAE (S$)",
            "cv_r2": "CV R²",
            "test_rmse": "Test RMSE (S$)",
            "test_mae": "Test MAE (S$)",
            "test_r2": "Test R²",
            # "fit_seconds": "Train time (s)",
            # "chosen": "Deployed",
            "fit_seconds": "Train time (s)",
            "n_trees": "Trees used",
            "chosen": "Deployed",
        }
    )
    st.dataframe(
        table.style.format(
            {
                "CV RMSE (S$)": "{:,.0f}",
                "CV MAE (S$)": "{:,.0f}",
                "CV R²": "{:.3f}",
                "Test RMSE (S$)": "{:,.0f}",
                "Test MAE (S$)": "{:,.0f}",
                "Test R²": "{:.3f}",
                # "Train time (s)": "{:.1f}",
                "Train time (s)": "{:.1f}",
                "Trees used": "{:,.0f}",                
            }
        )
    )
    st.write("**Test MAE by model** (lower is better)")
    st.bar_chart(table["Test MAE (S$)"], horizontal=True, x_label="S$", y_label="")
    st.caption(
        "RMSE punishes large errors more than MAE. R² is the share of price "
        "variation the model explains (1.0 = perfect)."
    )
