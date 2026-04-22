# 🚗 Vehicle Price Prediction  
### *A Reproducible and Leakage-Resistant Approach to Vehicle Price Prediction Using Cross-Validated Regression Models*

---

## 📌 Executive Summary

This project develops a fully reproducible, leakage-safe machine learning pipeline to predict used vehicle prices using real-world Australian marketplace data.

Unlike typical regression projects, this work emphasises:

- strict leakage prevention through pipeline-based design
- statistically robust model comparison using cross-validation
- feature engineering from semi-structured real-world data
- diagnostic evaluation beyond aggregate metrics
- pipeline validation via ablation experiments

The final system demonstrates strong predictive performance while exposing structural challenges in modelling high-value vehicles, highlighting the limitations of purely data-driven pricing models.

---

## 🎯 Problem Formulation

Vehicle pricing is a non-linear, high-dimensional regression problem characterised by:

- heterogeneous feature types (numerical, categorical, semi-structured)
- high-cardinality categorical variables (Brand, Model)
- skewed target distribution
- missing and noisy observations
- unobserved latent factors (e.g., condition, seller behaviour)

### Research Question

> How effectively can machine learning models predict vehicle prices from marketplace data, and which modelling approach offers the best balance between accuracy, robustness, and generalisation?

---

## 📊 Dataset

- **Source:** Kaggle – Australian Vehicle Prices Dataset
- **Size:** ~16,700 observations × 19 features
- **Type:** Structured + Semi-structured

### Feature Categories

| Type | Examples |
|------|----------|
| Numerical | Year, Price |
| Semi-structured | Engine, Kilometres, FuelConsumption |
| Categorical | Brand, Model, FuelType |
| Location | City, State |

---

## ⚠️ Real-World Data Challenges

- Missing values:
  - Seats (~10%)
  - Doors (~9.5%)
  - Location (~2.7%)
- High-cardinality categorical variables
- Semi-structured text fields requiring parsing
- Strong right-skew in target variable
- Non-random missingness patterns

---

## 📈 Exploratory Analysis

### Statistical Observations

- Raw price skewness: **8.66**
- Log-transformed skewness: **0.12**
- Key relationships:
  - Price increases with Year
  - Price decreases with Mileage

### Insight

> Vehicle pricing is driven by multi-variable interactions, not a single dominant feature.

---

## ⚙️ Methodological Framework

The system is designed to ensure:

- strict separation of training and evaluation
- no data leakage
- fair model comparison
- reproducibility

---

## 🔧 Preprocessing & Pipeline Design

### 🔐 Leakage-Safe Pipeline

All transformations are implemented using **scikit-learn Pipeline** and **ColumnTransformer**, ensuring:

- transformations are fitted only on training folds
- identical processing across cross-validation and test data
- no information leakage

### Pipeline Structure

```text
Raw Data
   ↓
Feature Engineering (custom transformer)
   ↓
ColumnTransformer:
   ├── Numerical → Median Imputation
   └── Categorical → One-Hot Encoding (handle_unknown='ignore')
   ↓
Model (Regressor)
````

---

## 🧠 Feature Engineering

A custom transformer extracts structured signals from raw marketplace fields.

### Key Derived Features

* `car_age`
* `kilometres_num`
* `kms_per_year`
* `engine_litres`
* `fuel_cons_l_per_100km`
* `doors_num`
* `seats_num`
* `cylinders_num`
* `litres_per_cylinder`
* `fuel_per_litre_engine`
* `city`
* `state`
* missing-value indicators

### Justification

* `kms_per_year` normalises mileage across vehicle age
* `engine_litres` captures vehicle performance class
* `litres_per_cylinder` acts as a proxy for engine efficiency
* location-derived features capture regional price variation

> Feature engineering converts raw marketplace noise into structured predictive signal.

---

## 🧪 Modelling Framework

### Models Evaluated

* Dummy Regressor (baseline)
* Ridge Regression
* Lasso Regression
* ElasticNet
* Random Forest
* XGBoost

---

## 🔍 Validation Strategy

* **Train/Test Split:** 80 / 20
* **Cross-validation:** 5-fold KFold (training only)
* **Hyperparameter tuning:** RandomizedSearchCV
* **Final evaluation:** locked test set

### Key Principle

> All preprocessing and feature engineering are fitted only within training folds.

---

## 📏 Evaluation Metrics

| Metric | Purpose                |
| ------ | ---------------------- |
| RMSE   | Penalises large errors |
| MAE    | Interpretable error    |
| R²     | Variance explained     |

---

## 🏆 Results

### Best Model: **XGBoost**

| Metric | Value          |
| ------ | -------------- |
| RMSE   | **13,700 AUD** |
| MAE    | **5,070 AUD**  |
| R²     | **0.842**      |

### Cross-Validation Performance

* RMSE (log): **0.2173**
* MAE (log): **0.1488**
* R² (log): **0.897**

### Interpretation

> On average, predictions are within approximately 5,000 AUD, which is strong given the variability of real-world marketplace pricing.

---

## 📊 Model Comparison

### Ranking

1. XGBoost
2. Random Forest
3. ElasticNet
4. Lasso
5. Ridge
6. Dummy Regressor

### Why XGBoost Performed Best

* captures non-linear relationships automatically
* handles sparse one-hot encoded features efficiently
* is robust to multicollinearity
* offers a strong bias-variance trade-off for tabular data

---

## 🔬 Diagnostic Analysis

### Residual Behaviour

* residuals are centred near zero, indicating broadly unbiased predictions
* residual variance increases with price, indicating heteroscedasticity

### Error by Price Segment

| Segment | MAE        |
| ------- | ---------- |
| Low     | ~2,230 AUD |
| Mid     | ~2,967 AUD |
| High    | ~9,870 AUD |

### Insight

> High-value vehicles exhibit substantially greater uncertainty, making them harder to predict accurately.

---

## 🧪 Ablation Study

| Configuration              | Effect                 |
| -------------------------- | ---------------------- |
| Remove feature engineering | Major performance drop |
| Remove log transform       | Increased error        |
| Remove location            | Moderate drop          |
| Remove title               | Minimal impact         |

### Conclusion

> Performance gains arise from the overall pipeline design, not just model choice alone.

---

## 🧱 Project Architecture

```text
project/
├── data/
│   └── Australian Vehicle Prices.csv
├── figures/
│   ├── EDA plots
│   ├── model comparison plots
│   └── diagnostic plots
├── outputs/
│   ├── final_model_comparison.csv
│   ├── test_set_results.csv
│   ├── cv_summary_results.csv
│   ├── diagnostics.csv
│   ├── ablation_results.csv
│   └── reproducibility_snapshot.json
├── Australian Vehicle Prices.ipynb
└── README.md
```

---

## 🔁 Reproducibility

* fixed random seed
* pipeline-based preprocessing
* controlled cross-validation
* exported outputs (CSV + JSON)
* environment snapshot included

---

## 💻 Computational Context

* Python
* pandas
* scikit-learn
* XGBoost
* CPU-based training
* efficient runtime suitable for standard hardware

---

## ⚠️ Limitations

* missing key features such as condition and service history
* no temporal modelling (static snapshot of listings)
* high-end vehicles are harder to predict accurately
* ensemble models are less interpretable than simpler linear baselines
* possible distribution shift in real-world deployment

---

## 🔮 Future Work

* SHAP-based explainability
* temporal price modelling
* richer vehicle-condition and seller metadata
* LightGBM / CatBoost comparison
* deployment as an API or dashboard

---

## 💡 Key Takeaways

* ensemble methods outperform linear models for this tabular pricing problem
* feature engineering is critical when working with messy real-world data
* proper validation is essential to avoid misleading performance estimates
* real-world vehicle pricing remains inherently uncertain, especially in premium segments

---

## 👨‍💻 Author

**Mahesh Babu Poka**
MSc Data Science – University of Hertfordshire

---

## 📌 Final Insight

> This project demonstrates that robust methodology, rigorous validation, and thoughtful feature engineering are more important than model choice alone in real-world machine learning systems.

