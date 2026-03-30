# Shark Attack Prediction Project

## Overview
This project analyzes historical shark attack data and builds machine learning models to predict whether an attack is fatal.

The goal is to:
- Clean and preprocess real-world messy data
- Engineer meaningful features (activity, species, time, season, etc.)
- Train multiple ML models
- Evaluate model performance and compare results

---

## Research Paper

This project resulted in a research paper submitted to the Curetrix Journal.

📄 [Read the full paper](docs/paper/shark_attack_prediction_ml_analysis.pdf)

### Summary
- Developed ML models to predict shark attack fatality
- Engineered domain-specific features (activity, species, time, season)
- Compared multiple classifiers with PR-AUC evaluation

---

## Project Structure
```
project/
├── data/
│   └── raw/attacks_cleaned.csv
├── notebooks/
│   ├── 01_eda_refactored.ipynb
│   └── 02_modeling_refactored.ipynb
├── src/
│   ├── data.py
│   ├── mappings.py
│   ├── features.py
│   ├── preprocessing.py
│   └── modeling.py
├── docs/
│   └── paper/
│       └── shark_attack_prediction_ml_analysis.pdf
├── requirements.txt
└── README.md
```

---

## Dataset
- Source: Shark attack dataset
- Contains:
  - Location
  - Activity during attack
  - Shark species
  - Victim demographics
  - Time and date

---

## Key Features Engineered
- Activity categories (swimming, surfing, diving, etc.)
- Shark species classification
- Time of day
- Season
- Hemisphere and ocean mapping
- Weekend indicator

---

## Models Used
- Logistic Regression
- Decision Tree
- Random Forest
- KNN
- XGBoost
- LightGBM

---

## Evaluation Metrics
- Accuracy
- Precision / Recall / F1
- PR-AUC (important for imbalanced datasets)

---

## How to Run

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Run notebooks:
- Start with EDA
- Then modeling

---

## Future Improvements
- Better handling of missing data
- Advanced feature engineering
- Deep learning approaches
- Real-time prediction system

---

## Author
High School ML Research Project
