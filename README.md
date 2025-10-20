# California House Price Predictor

Predict California residential **close prices** using an **XGBoost** model trained on CRMLS sales data. Includes a Streamlit app for interactive inference and a notebook + script for training.

## Features

* Clean, reproducible preprocessing for CRMLS exports.
* Target rounding to nearest $1,000 for stability.
* ZIP→quantile **GeoCluster** feature engineered from training medians.
* Robust imputations for numeric and categorical fields.
* Outlier trimming and one-hot alignment between train/test.
* Streamlit UI with normal-curve distribution plot for the selected city and a marker at the predicted price.

---

## Repo structure

```
.
├── app.py                     # Streamlit app
├── training_notebook.ipynb    # Your EDA/training notebook (example name)
├── requirements.txt
├── runtime.txt                # python-3.11 recommended
├── xgb_houseprice.pkl         # model (artifact, created by training)
├── model_columns.json         # feature list used by the app (artifact)
├── pc2cluster.pkl             # PostalCode→GeoCluster map (artifact)
├── eval_df.pkl                # City, Actual, Predicted (artifact, optional)
└── data/                      # put CSVs here (ignored by app)
    ├── CRMLSSold202501_filled.csv
    ├── ...
    └── CRMLSSold202509.csv
```

---

## Data

**Input:** Monthly CRMLS sold CSVs (2025-01 … 2025-09).
**Filters:**

* `PropertyType == "Residential"` and `PropertySubType == "SingleFamilyResidence"`.
* Drop high-leakage/ID/contact columns.
* Remove non-CA/US rows and unify city aliases (e.g., East/Lake Los Angeles → Los Angeles; Big Bear*, Carmel-By-The-Sea → Carmel; etc.).
* Round `ClosePrice` to nearest $1,000.

**Missing data:**

* Boolean cols (`ViewYN`, `PoolPrivateYN`, `AttachedGarageYN`, `FireplaceYN`, `NewConstructionYN`) → `False`.
* `LivingArea`, `AssociationFee` → `RobustScaler` + `KNNImputer(k=7)`, train-fit then apply to test.
* `Stories` → most-frequent imputer; derive `Levels` from `Stories` with fallback to train mode.
* Selected numeric cols → median imputer → round → `Int64`.

**Train/Test split:**

* `CloseDate` in **2025-09** → **test**; everything else → **train**.

**GeoCluster:**

* Compute per-ZIP median `ClosePrice` on **train only**.
* Bin into **k quantiles** (default `k=10`, reduced if uniques < k).
* Map to `GeoCluster` integers; one-hot with prefix `GC_`.

---

## Model

* **XGBRegressor**

  ```
  n_estimators=600
  learning_rate=0.05
  max_depth=8
  subsample=0.8
  colsample_bytree=0.8
  tree_method='hist'
  random_state=42
  ```
* **Features kept:**

  * Core numerics: `LivingArea, ParkingTotal, LotSizeAcres, BathroomsTotalInteger, BedroomsTotal, Stories, MainLevelBedrooms, GarageSpaces, Age`
  * Geography: `City` (one-hot)
  * Engineered: `GC_*` dummies
* **Outliers:** trim `ClosePrice` at 0.5% tails in train and test.
* **Alignment:** one-hot on train; reindex test to train columns.

**Metrics (example from your logs):**

* `R² ≈ 0.86`
* `MAPE ≈ 0.156`
* `MdAPE ≈ 10.98%`

---

## Streamlit app

**What it does:**
Takes user inputs, builds feature row, encodes `City` and `GeoCluster` from the provided `PostalCode`, predicts price, **rounds to $1,000**, and shows a **histogram + normal curve** of actual city prices with a vertical line at the prediction.

**Run locally:**

```bash
pip install -r requirements.txt
streamlit run app.py
```

**Artifacts required in repo root (created by training):**

* `xgb_houseprice.pkl`
* `model_columns.json`
* `pc2cluster.pkl`
* `eval_df.pkl` (optional, for plots)

**Deployment (Streamlit Community Cloud):**
Push code + artifacts to GitHub → “New app” → select repo/branch → `app.py`.
Set Python via `runtime.txt`:

```
python-3.11
```

---

## Training pipeline (script/notebook outline)

1. **Load monthly CSVs** and `pd.concat`.
2. **Filter property types** and **drop leakage/ID columns**.
3. **Normalize cities** and **remove outside CA/US**.
4. **Round `ClosePrice`**, handle **missing values** as above.
5. **Split train/test** by `CloseDate` (2025-09 test).
6. **Impute** numerics and categories; derive `Levels`; cast types.
7. **Engineer GeoCluster** from train ZIP medians; one-hot `GC_*`.
8. **Select features** and **trim outliers**.
9. **One-hot** categoricals; align columns.
10. **Train XGBRegressor**; evaluate `R²`, `MAPE`, `MdAPE`.
11. **Export artifacts**:

    ```python
    pickle.dump(model, open("xgb_houseprice.pkl","wb"))
    json.dump({"columns": list(X_train.columns)}, open("model_columns.json","w"))
    pickle.dump(PC2CLUSTER, open("pc2cluster.pkl","wb"))
    # Optional eval for app plots:
    eval_df = test_t[["City"]].copy()
    eval_df["Actual"] = y_test.values
    eval_df["Predicted"] = y_pred
    pickle.dump(eval_df, open("eval_df.pkl","wb"))
    ```

---

## Requirements

Minimal set (pin to your working versions to avoid pickle drift):

```
streamlit
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
scipy
plotly
```

Recommended: pin exact versions and add `runtime.txt` with `python-3.11`.

---

## License and data usage

CRMLS data is licensed. Ensure your use and redistribution comply with CRMLS terms. Do not commit raw proprietary data to public repos.

---

## Quick commands

```bash
# format + lint (optional if you set up tools)
pip install ruff black
ruff check .
black .

# run app
streamlit run app.py

# retrain artifacts (run your notebook or script), then:
git add xgb_houseprice.pkl model_columns.json pc2cluster.pkl eval_df.pkl
git commit -m "Update model artifacts"
git push
```

