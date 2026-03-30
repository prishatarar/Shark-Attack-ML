# Project Summary: Shark Attack Fatality Prediction

## Problem
Shark attacks are rare but high-impact events. Understanding the factors that influence whether an attack becomes fatal can help improve awareness, safety measures, and research in marine environments.

This project aims to predict whether a shark attack is fatal using historical data and machine learning.

---

## Approach

### Data
- Historical shark attack dataset
- Includes features such as:
  - Location (country, ocean, hemisphere)
  - Activity during attack
  - Shark species
  - Victim demographics (age, sex)
  - Time and date

---

### Feature Engineering
Significant effort was spent transforming raw, messy data into meaningful features:

- Activity classification (e.g., swimming, surfing, diving)
- Shark species grouping (e.g., dangerous vs other)
- Time of day categorization
- Season extraction from date
- Geographic mapping (country → ocean, hemisphere)
- Weekend indicator

---

### Modeling
Multiple machine learning models were trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- XGBoost
- LightGBM

---

### Evaluation
Due to class imbalance, models were evaluated using:
- Precision, Recall, F1-score
- PR-AUC (Primary metric)

---

## Key Results
*(Update these after running your models)*

- Best Model: Random Forest (example)
- PR-AUC: XX
- Accuracy: XX%

### Insights:
- Certain activities (e.g., surfing/swimming) show higher fatality correlation
- Specific shark species are more associated with fatal outcomes
- Time of day and season have measurable impact

---

## Why This Matters
This project demonstrates how machine learning can extract meaningful patterns from noisy real-world data and contribute to understanding rare but critical events.

It also showcases:
- End-to-end ML pipeline design
- Feature engineering from unstructured data
- Model comparison under class imbalance

---

## Research Output
This work resulted in a research paper:

📄 shark_attack_prediction_ml_analysis.pdf

---

## Future Work
- Incorporate environmental factors (water temperature, depth)
- Improve species classification with external datasets
- Explore deep learning approaches
- Build real-time prediction system

---

## Author
High School Student Research Project (AI + Oceanography)
