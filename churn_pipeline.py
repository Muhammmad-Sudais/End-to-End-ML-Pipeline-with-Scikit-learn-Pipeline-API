"""
ML Pipeline for Customer Churn Prediction
This script implements a complete ML pipeline with preprocessing, model training,
hyperparameter tuning using GridSearchCV, and model evaluation.
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def load_data():
    """
    Load the Telco Churn dataset.
    """
    print("="*80)
    print("LOADING DATASET")
    print("="*80)
    
    df = pd.read_csv('data/telco_churn.csv')
    print(f"✓ Dataset loaded successfully! Shape: {df.shape}")
    
    return df

def prepare_data(df):
    """
    Prepare data for modeling by separating features and target.
    """
    print("\n" + "="*80)
    print("PREPARING DATA")
    print("="*80)
    
    # Handle TotalCharges - convert to numeric (some values might be spaces)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Drop customerID as it's not a feature
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Identify numerical and categorical columns
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\n✓ Features: {X.shape[1]}")
    print(f"  - Numerical: {len(numerical_features)}")
    print(f"  - Categorical: {len(categorical_features)}")
    print(f"✓ Target variable: Churn (Binary: 0=No, 1=Yes)")
    print(f"  - Class distribution: {y.value_counts().to_dict()}")
    
    return X, y, numerical_features, categorical_features

def create_preprocessing_pipeline(numerical_features, categorical_features):
    """
    Create preprocessing pipeline for numerical and categorical features.
    """
    print("\n" + "="*80)
    print("CREATING PREPROCESSING PIPELINE")
    print("="*80)
    
    # Numerical pipeline: impute missing values + scale
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: impute missing values + one-hot encode
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine both pipelines
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])
    
    print("✓ Preprocessing pipeline created:")
    print("  - Numerical: SimpleImputer(median) → StandardScaler")
    print("  - Categorical: SimpleImputer(constant) → OneHotEncoder")
    
    return preprocessor

def train_logistic_regression(X_train, y_train, X_test, y_test, preprocessor):
    """
    Train Logistic Regression with GridSearchCV for hyperparameter tuning.
    """
    print("\n" + "="*80)
    print("TRAINING LOGISTIC REGRESSION")
    print("="*80)
    
    # Create pipeline
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    # Define hyperparameter grid
    param_grid = {
        'classifier__C': [0.01, 0.1, 1, 10, 100],
        'classifier__penalty': ['l2'],
        'classifier__solver': ['lbfgs', 'liblinear']
    }
    
    print("\n🔍 Performing GridSearchCV...")
    print(f"   Parameter grid: {param_grid}")
    
    # GridSearchCV
    grid_search = GridSearchCV(
        lr_pipeline,
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n✓ Best parameters: {grid_search.best_params_}")
    print(f"✓ Best CV ROC-AUC score: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    y_pred = grid_search.predict(X_test)
    y_pred_proba = grid_search.predict_proba(X_test)[:, 1]
    
    return grid_search, y_pred, y_pred_proba

def train_random_forest(X_train, y_train, X_test, y_test, preprocessor):
    """
    Train Random Forest with GridSearchCV for hyperparameter tuning.
    """
    print("\n" + "="*80)
    print("TRAINING RANDOM FOREST")
    print("="*80)
    
    # Create pipeline
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Define hyperparameter grid
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [10, 20, 30, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }
    
    print("\n🔍 Performing GridSearchCV...")
    print(f"   Parameter grid: {param_grid}")
    
    # GridSearchCV
    grid_search = GridSearchCV(
        rf_pipeline,
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n✓ Best parameters: {grid_search.best_params_}")
    print(f"✓ Best CV ROC-AUC score: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    y_pred = grid_search.predict(X_test)
    y_pred_proba = grid_search.predict_proba(X_test)[:, 1]
    
    return grid_search, y_pred, y_pred_proba

def evaluate_model(model_name, y_test, y_pred, y_pred_proba):
    """
    Evaluate model performance with multiple metrics.
    """
    print("\n" + "-"*80)
    print(f"MODEL EVALUATION: {model_name}")
    print("-"*80)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n📊 Performance Metrics:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   ROC-AUC:   {roc_auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📈 Confusion Matrix:")
    print(f"   TN: {cm[0,0]:4d}  |  FP: {cm[0,1]:4d}")
    print(f"   FN: {cm[1,0]:4d}  |  TP: {cm[1,1]:4d}")
    
    # Classification Report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }

def visualize_results(lr_metrics, rf_metrics):
    """
    Create visualizations comparing model performance.
    """
    os.makedirs('results', exist_ok=True)
    
    # 1. Metrics comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    lr_values = [lr_metrics[m] for m in metrics]
    rf_values = [rf_metrics[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, lr_values, width, label='Logistic Regression', color='#3498db')
    bars2 = ax.bar(x + width/2, rf_values, width, label='Random Forest', color='#2ecc71')
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper().replace('_', '-') for m in metrics])
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✓ Saved: results/model_comparison.png")
    
    # 2. Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Logistic Regression
    sns.heatmap(lr_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'],
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title('Logistic Regression\nConfusion Matrix', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Actual', fontsize=11)
    axes[0].set_xlabel('Predicted', fontsize=11)
    
    # Random Forest
    sns.heatmap(rf_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Greens',
                xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'],
                ax=axes[1], cbar_kws={'label': 'Count'})
    axes[1].set_title('Random Forest\nConfusion Matrix', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Actual', fontsize=11)
    axes[1].set_xlabel('Predicted', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('results/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: results/confusion_matrices.png")

def save_best_model(lr_model, rf_model, lr_metrics, rf_metrics):
    """
    Save the best performing model.
    """
    os.makedirs('models', exist_ok=True)
    
    print("\n" + "="*80)
    print("SAVING MODELS")
    print("="*80)
    
    # Compare ROC-AUC scores
    if lr_metrics['roc_auc'] > rf_metrics['roc_auc']:
        best_model = lr_model
        best_model_name = 'Logistic Regression'
        best_score = lr_metrics['roc_auc']
    else:
        best_model = rf_model
        best_model_name = 'Random Forest'
        best_score = rf_metrics['roc_auc']
    
    # Save best model
    joblib.dump(best_model, 'models/best_churn_pipeline.pkl')
    print(f"\n✓ Best model saved: {best_model_name}")
    print(f"  ROC-AUC Score: {best_score:.4f}")
    print(f"  Location: models/best_churn_pipeline.pkl")
    
    # Also save both models
    joblib.dump(lr_model, 'models/logistic_regression_pipeline.pkl')
    joblib.dump(rf_model, 'models/random_forest_pipeline.pkl')
    print(f"\n✓ All models saved:")
    print(f"  - models/logistic_regression_pipeline.pkl")
    print(f"  - models/random_forest_pipeline.pkl")
    print(f"  - models/best_churn_pipeline.pkl")

def main():
    """
    Main function to orchestrate the ML pipeline.
    """
    print("="*80)
    print("CUSTOMER CHURN PREDICTION - ML PIPELINE")
    print("="*80)
    
    # Load data
    df = load_data()
    
    # Prepare data
    X, y, numerical_features, categorical_features = prepare_data(df)
    
    # Train-test split
    print("\n" + "="*80)
    print("SPLITTING DATA")
    print("="*80)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Training set: {X_train.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples")
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(numerical_features, categorical_features)
    
    # Train Logistic Regression
    lr_model, lr_pred, lr_pred_proba = train_logistic_regression(
        X_train, y_train, X_test, y_test, preprocessor
    )
    lr_metrics = evaluate_model('Logistic Regression', y_test, lr_pred, lr_pred_proba)
    
    # Train Random Forest
    rf_model, rf_pred, rf_pred_proba = train_random_forest(
        X_train, y_train, X_test, y_test, preprocessor
    )
    rf_metrics = evaluate_model('Random Forest', y_test, rf_pred, rf_pred_proba)
    
    # Visualize results
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    visualize_results(lr_metrics, rf_metrics)
    
    # Save best model
    save_best_model(lr_model, rf_model, lr_metrics, rf_metrics)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print("\nNext Steps:")
    print("1. Review model performance in 'results/' directory")
    print("2. Use 'predict.py' to make predictions on new data")
    print("3. Deploy the saved pipeline to production")

if __name__ == "__main__":
    main()
