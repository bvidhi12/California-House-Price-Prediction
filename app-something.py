import streamlit as st
import numpy as np
import pandas as pd
import pickle

# load models
xgb_model = pickle.load(open("xgb_model.pkl","rb"))
linreg_model = pickle.load(open("linreg_model.pkl","rb"))
train_cols = pickle.load(open("train_cols.pkl","rb"))

st.title("🏠 House Price Prediction")

# ---- inputs
LivingArea = st.number_input("Living Area (sqft)", 200, 10000, 1500, 50)
BedroomsTotal = st.number_input("Bedrooms", 0, 12, 3, 1)
BathroomsTotalInteger = st.number_input("Bathrooms", 0.0, 12.0, 2.0, 0.5)
ParkingTotal = st.number_input("Total Parking", 0, 10, 2, 1)
GarageSpaces = st.number_input("Garage Spaces", 0, 10, 1, 1)
PostalCode = st.text_input("Postal Code", "12345")

model_choice = st.selectbox("Model", ["Linear Regression","XGBoost"])

def make_X():
    df = pd.DataFrame([{
        "LivingArea": LivingArea,
        "ParkingTotal": ParkingTotal,
        "BathroomsTotalInteger": BathroomsTotalInteger,
        "BedroomsTotal": BedroomsTotal,
        "GarageSpaces": GarageSpaces,
        "PostalCode": PostalCode
    }])

    one_hot_pc = pd.get_dummies(df["PostalCode"].astype(str), prefix="PostalCode", dtype=int)
    X = pd.concat([df[["LivingArea","ParkingTotal","BathroomsTotalInteger","BedroomsTotal","GarageSpaces"]],
                   one_hot_pc], axis=1)

    X["LivingArea"] = X["LivingArea"]**0.7
    X["BathroomsTotalInteger"] = X["BathroomsTotalInteger"]**1.2

    X = X.reindex(columns=train_cols, fill_value=0)
    return X

if st.button("Predict"):
    X = make_X()
    if model_choice == "Linear Regression":
        yhat = np.expm1(linreg_model.predict(X))
    else:
        yhat = np.expm1(xgb_model.predict(X))
    st.success(f"Predicted Price: ${yhat[0]:,.0f}")
