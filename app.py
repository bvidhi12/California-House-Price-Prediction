# app.py
import pickle
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go


# ---------- load artifacts ----------
@st.cache_resource(show_spinner=False)
@st.cache_resource(show_spinner=False)
def load_artifacts():
    with open("xgb_houseprice.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model_columns.json", "r") as f:
        feature_cols = json.load(f)["columns"]
    with open("pc2cluster.pkl", "rb") as f:
        pc2cluster = pickle.load(f)
    try:
        eval_df = pickle.load(open("eval_df.pkl","rb"))  # columns: City, Actual, Predicted
    except Exception:
        eval_df = None
    return model, feature_cols, pc2cluster, eval_df

model, FEATURE_COLS, PC2CLUSTER, EVAL_DF = load_artifacts()


model, FEATURE_COLS, PC2CLUSTER, CITY_POSTALS = load_artifacts()

# ---------- helpers ----------
def build_gc_vector(postal_code, feature_cols, pc2cluster):
    gc_cols = [c for c in feature_cols if c.startswith("GC_")]
    row = {c: 0 for c in gc_cols}
    if not postal_code:
        return row, None

    label = pc2cluster.get(str(postal_code).strip().upper(), -1)
    if label == -1:
        return row, -1

    # handle either "GC_6" or "GC_6_True" naming
    for c in (f"GC_{label}", f"GC_{label}_True"):
        if c in gc_cols:
            row[c] = 1
            return row, label

    # baseline dummy dropped during training
    return row, label

def build_city_vector(city, feature_columns):
    city_cols = [c for c in feature_columns if c.startswith("City_")]
    row = {c: 0 for c in city_cols}
    if city:
        key = f"City_{city}"
        if key in city_cols:
            row[key] = 1
    return row

def assemble_row(inputs, feature_columns):
    row = {
        "LivingArea": float(inputs["LivingArea"]),
        "ParkingTotal": int(inputs["ParkingTotal"]),
        "LotSizeAcres": float(inputs["LotSizeAcres"]),
        "BathroomsTotalInteger": int(inputs["BathroomsTotalInteger"]),
        "BedroomsTotal": int(inputs["BedroomsTotal"]),
        "Stories": float(inputs["Stories"]),
        "MainLevelBedrooms": int(inputs["MainLevelBedrooms"]),
        "GarageSpaces": int(inputs["GarageSpaces"]),
        "Age": float(inputs["Age"]),
    }
    row.update(build_city_vector(inputs["City"], FEATURE_COLS))
    gc_row, _ = build_gc_vector(inputs["PostalCode"], FEATURE_COLS, PC2CLUSTER)
    row.update(gc_row)
    X = pd.DataFrame([row], columns=feature_columns).fillna(0)
    return X

# ---------- UI ----------
st.set_page_config(page_title="California House Price Predictor", layout="centered")
st.title("California House Price Predictor")

st.subheader("Please Input your house details")

# cities seen during training (from dummy columns)
TRAIN_CITIES = sorted({c.replace("City_", "") for c in FEATURE_COLS if c.startswith("City_")})

# ---- Input order (as requested) ----
# 1. Total Bedrooms
bedrooms_total = st.number_input("Total Bedrooms", min_value=0, max_value=20, value=3, step=1)

# 2. Main Level Bedrooms
main_level_bedrooms = st.number_input("Main Level Bedrooms", min_value=0, max_value=10, value=1, step=1)

# 3. Total Bathrooms (integer)
bathrooms_total_integer = st.number_input("Total Bathrooms", min_value=0, max_value=20, value=2, step=1, format="%d")

# 4. Living Area (sqft)
living_area = st.number_input("Living Area (sqft)", min_value=100.0, max_value=20000.0, value=2000.0, step=50.0)

# 5. Lot Size (acres)
lot_size_acres = st.number_input("Lot Size (acres)", min_value=0.0, max_value=100.0, value=0.12, step=0.01)

# 6. Bulit in year
year_built = st.number_input("Bulit in year", min_value=1850, max_value=2025, value=1995, step=1)
age = 2025 - year_built
st.caption(f"Computed Age: {age} years")

# 7. Total Parking Space
parking_total = st.number_input("Total Parking Space", min_value=0, max_value=20, value=2, step=1)

# 8. Garage Spaces
garage_spaces = st.number_input("Garage Spaces", min_value=0, max_value=10, value=2, step=1)

# 9. City
city = st.selectbox("City", options=["<unseen>"] + TRAIN_CITIES, index=0, help="Must match a training city to one-hot encode")

# 10. Postal COde (with city-based suggestions)
postal_suggestions = CITY_POSTALS.get(city, []) if city and city != "<unseen>" else []
if postal_suggestions:
    st.markdown(
        "**Postal codes seen for this city:** " + ", ".join(f"`{z}`" for z in postal_suggestions)
    )
postal = st.text_input("Postal COde", value="", help="Used to derive GeoCluster")

# Predict button
if st.button("Predict", type="primary"):
    if not postal.strip():
        st.error("PostalCode is required to derive GeoCluster.")
    else:
        inputs = {
            "LivingArea": living_area,
            "ParkingTotal": parking_total,
            "LotSizeAcres": lot_size_acres,
            "BathroomsTotalInteger": bathrooms_total_integer,
            "BedroomsTotal": bedrooms_total,
            "Stories": 1.0,  # if you want this exposed, add another input; else keep default
            "MainLevelBedrooms": main_level_bedrooms,
            "GarageSpaces": garage_spaces,
            "Age": age,
            "City": None if city == "<unseen>" else city,
            "PostalCode": postal,
        }
        X = assemble_row(inputs, FEATURE_COLS)
        pred = float(model.predict(X)[0])

        st.success(f"Your predicted house price is ${pred:,.0f}")

        # after st.success(f"Your predicted house price is ${pred:,.0f}")
        if inputs["City"] and EVAL_DF is not None:
            sub = EVAL_DF.loc[EVAL_DF['City'] == inputs["City"]].copy()
            if not sub.empty:
                sub = sub.sort_values('Actual').reset_index(drop=True)
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=sub['Actual'], mode='lines', name='Actual ClosePrice'))
                fig.add_trace(go.Scatter(y=sub['Predicted'], mode='lines', name='Predicted ClosePrice'))
                fig.update_layout(
                    title=f"Actual vs Predicted — {inputs['City']}",
                    xaxis_title="Samples (sorted by actual value)",
                    yaxis_title="ClosePrice",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No evaluation rows for this city in eval_df.")
        else:
            st.info("City comparison plot unavailable. Missing city or eval_df.pkl.")


        # Diagnostics
        active_gc = [c for c in X.columns if c.startswith("GC_") and X.iloc[0][c] == 1]
        st.caption(f"GeoCluster dummy: {active_gc[0] if active_gc else 'baseline (unknown cluster)'}")
        if inputs["City"] and f"City_{inputs['City']}" not in FEATURE_COLS:
            st.warning("City unseen in training. Treated as baseline for City.")
