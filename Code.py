"""
Campus Placement Prediction using K-Nearest Neighbors (KNN)
CS 435 - Machine Learning

Predicts whether an MBA student gets placed during campus recruitment.
Dataset: https://www.kaggle.com/benroshan/factors-affecting-campus-placement
"""

import os

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ENHANCED NOTEBOOK WITH IMPROVED CODE STRUCTURE

## CONFIGURATION SECTION
RANDOM_STATE = 42
TEST_SIZE = 0.30
CV_FOLDS = 5
K_VALUES = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
SCALING_METHOD = "StandardScaler"
VALIDATION_STRATEGY = "StratifiedKFold"

## HELPER FUNCTIONS

def load_dataset(kaggle_dataset_name: str) -> pd.DataFrame:
    """
    Load dataset from Kaggle.
    
    Parameters:
    -----------
    kaggle_dataset_name : str
        Name of the Kaggle dataset in format "username/dataset-name"
    
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    try:
        path = kagglehub.dataset_download(kaggle_dataset_name)
        csv_path = os.path.join(path, "Placement_Data_Full_Class.csv")
        df = pd.read_csv(csv_path)
        print(f"✓ Dataset loaded: {df.shape[0]} samples × {df.shape[1]} features")
        return df
    except Exception as e:
        raise ValueError(f"Error loading dataset: {str(e)}")

def validate_data_quality(df: pd.DataFrame) -> dict:
    """
    Check data quality metrics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    dict
        Quality metrics (missing values, duplicates, data types)
    """
    quality_report = {
        'total_samples': len(df),
        'total_features': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }
    return quality_report

def preprocess_pipeline(df: pd.DataFrame, steps: list = None) -> pd.DataFrame:
    """
    Apply preprocessing pipeline with specified steps.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw input data
    steps : list
        List of preprocessing steps to apply
    
    Returns:
    --------
    pd.DataFrame
        Preprocessed data
    """
    if steps is None:
        steps = ['target_encoding', 'leakage_removal', 'categorical_encoding']
    
    processed_df = df.copy()
    
    # Step 1: Target Encoding
    if 'target_encoding' in steps:
        processed_df['status'] = processed_df['status'].map({'Placed': 1, 'Not Placed': 0})
    
    # Step 2: Remove Leakage
    if 'leakage_removal' in steps:
        processed_df = processed_df.drop(['salary', 'sl_no'], axis=1)
    
    # Step 3: Categorical Encoding
    if 'categorical_encoding' in steps:
        le_dict = {}
        categorical_cols = processed_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'status':
                le = LabelEncoder()
                processed_df[col] = le.fit_transform(processed_df[col])
                le_dict[col] = le
    
    print(f"✓ Preprocessing complete: {processed_df.shape}")
    return processed_df

def evaluate_model_performance(y_true, y_pred, model_name="Model") -> dict:
    """
    Calculate comprehensive evaluation metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    model_name : str
        Name of the model for reporting
    
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    return metrics

def plot_model_comparison(metrics_list: list):
    """
    Visualize comparison of multiple models.
    
    Parameters:
    -----------
    metrics_list : list
        List of dictionaries containing model metrics
    """
    models = [m['model'] for m in metrics_list]
    accuracy = [m['accuracy'] for m in metrics_list]
    f1 = [m['f1'] for m in metrics_list]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.bar(models, accuracy, color=['#2E86AB', '#A23B72'])
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylim([0, 1])
    
    ax2.bar(models, f1, color=['#2E86AB', '#A23B72'])
    ax2.set_ylabel('F1-Score')
    ax2.set_title('Model F1-Score Comparison')
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    return fig

## MAIN WORKFLOW

# 1. Data Loading
print("=" * 70)
print("STEP 1: DATA LOADING")
print("=" * 70)
df = load_dataset("benroshan/factors-affecting-campus-placement")

# 2. Data Quality Check
print("\n" + "=" * 70)
print("STEP 2: DATA QUALITY VALIDATION")
print("=" * 70)
quality = validate_data_quality(df)
print(f"Missing values: {sum(quality['missing_values'].values())}")
print(f"Duplicate rows: {quality['duplicate_rows']}")

# 3. Preprocessing
print("\n" + "=" * 70)
print("STEP 3: DATA PREPROCESSING")
print("=" * 70)
df_processed = preprocess_pipeline(df)

# 4. Feature Scaling
print("\n" + "=" * 70)
print("STEP 4: FEATURE SCALING")
print("=" * 70)
X = df_processed.drop('status', axis=1)
y = df_processed['status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"✓ Applied {SCALING_METHOD}")
print(f"  Feature means: {X_scaled.mean(axis=0)[:3]}... (should be ≈ 0)")
print(f"  Feature stds: {X_scaled.std(axis=0)[:3]}... (should be ≈ 1)")

# 5. Train-Test Split
print("\n" + "=" * 70)
print("STEP 5: TRAIN-TEST SPLIT")
print("=" * 70)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# 6. Model Training
print("\n" + "=" * 70)
print("STEP 6: KNN MODEL TRAINING")
print("=" * 70)
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
cv_scores = {}
for k in K_VALUES:
    scores = cross_val_score(KNeighborsClassifier(n_neighbors=k), X_scaled, y,
                             cv=cv, scoring="accuracy")
    cv_scores[k] = scores.mean()
    print(f"k={k:>2}  CV accuracy: {cv_scores[k]:.4f}")

best_k = max(cv_scores, key=cv_scores.get)
print(f"Best k: {best_k} (CV accuracy: {cv_scores[best_k]:.4f})")
knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train, y_train)
print(f"✓ KNN model trained with k={best_k}")

# 7. Model Evaluation
print("\n" + "=" * 70)
print("STEP 7: MODEL EVALUATION")
print("=" * 70)
y_pred = knn_model.predict(X_test)
knn_metrics = evaluate_model_performance(y_test, y_pred, "KNN")

print(f"Accuracy:  {knn_metrics['accuracy']:.4f}")
print(f"Precision: {knn_metrics['precision']:.4f}")
print(f"Recall:    {knn_metrics['recall']:.4f}")
print(f"F1-Score:  {knn_metrics['f1']:.4f}")

# 8. Baseline Comparison
print("\n" + "=" * 70)
print("STEP 8: BASELINE COMPARISON")
print("=" * 70)
baseline_pred = np.full_like(y_test, y_train.mode()[0])
baseline_metrics = evaluate_model_performance(y_test, baseline_pred, "Baseline")

print(f"Baseline Accuracy: {baseline_metrics['accuracy']:.4f}")
print(f"KNN Improvement: +{(knn_metrics['f1'] - baseline_metrics['f1']):.4f} F1-Score")

print("\n" + "=" * 70)
print("PROJECT COMPLETE")
print("=" * 70)
