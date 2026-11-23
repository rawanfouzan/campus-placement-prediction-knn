# Campus Placement Prediction using KNN

Predicting whether an MBA student will be placed during campus recruitment, based on their academic profile, using a **K-Nearest Neighbors (KNN)** classifier.

> Course project for **CS 435 – Machine Learning**, Department of Computer Science.

## Overview

Early identification of students who are less likely to be placed helps institutions offer targeted career counseling. This project builds and evaluates a KNN classifier on the *Factors Affecting Campus Placement* dataset and compares it against a naive majority-class baseline.

## Dataset

- **Source:** [Factors Affecting Campus Placement (Kaggle)](https://www.kaggle.com/benroshan/factors-affecting-campus-placement)
- **Size:** 215 MBA students
- **Features:** 12 after cleaning (10 continuous, 2 categorical), such as secondary/higher-secondary/degree/MBA percentages, employability test score, work experience, and specialisation
- **Target:** `status` (Placed = 1, Not Placed = 0)
- **Class balance:** ~68% Placed / ~32% Not Placed
- **Missing values:** none

## Methodology

1. **Target encoding:** `Placed` / `Not Placed` converted to 1 / 0
2. **Leakage removal:** dropped `salary` (only exists for placed students) and `sl_no` (row ID)
3. **Categorical encoding:** label encoding
4. **Feature scaling:** `StandardScaler` (essential for distance-based KNN)
5. **Split:** 70% train / 30% test, stratified, `random_state=42`
6. **Tuning:** k searched over 1–19 (odd values) with 5-fold stratified cross-validation, best **k = 5**
7. **Evaluation:** accuracy, precision, recall, F1, confusion matrix, and comparison with a baseline

## Results

| Metric | Score |
| --- | --- |
| CV accuracy (k = 5) | 83.72% |
| Test accuracy | 79.07% |
| Precision (weighted) | 78.25% |
| Recall (weighted) | 79.07% |
| F1-score (weighted) | 78.20% |

**Confusion matrix (test set)**

|  | Predicted Not Placed | Predicted Placed |
| --- | --- | --- |
| **Actual Not Placed** | 7 | 6 |
| **Actual Placed** | 3 | 27 |

The model identifies placed students well (90% recall) but is weaker on not-placed students (53.8% recall), which is expected given the class imbalance.

## Repository Structure

```
.
├── Code.py                # Full pipeline: loading, preprocessing, training, evaluation
├── report.docx            # Detailed project report
├── Presentation.pptx      # Project presentation slides
└── README.md
```

## Getting Started

**Requirements:** Python 3.8+

```bash
pip install pandas numpy scikit-learn matplotlib kagglehub
```

**Run:**

```bash
python Code.py
```

The dataset is downloaded automatically through `kagglehub` (a Kaggle account/API setup may be required).

## Future Work

- Handle class imbalance (e.g., SMOTE)
- Try ensemble methods and other distance metrics
- Feature selection / PCA
- Expand the dataset
- Deploy as a web app or API

## Team

- Aljazi Alauqab
- Jana Almishali
- Rawan Alfouzan

## Acknowledgments

Dataset by [Ben Roshan](https://www.kaggle.com/benroshan) on Kaggle, our course instructor, and the open-source Python libraries used.
