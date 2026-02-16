"""
Data Exploration Script for Telco Churn Dataset
This script downloads, loads, and explores the Telco Customer Churn dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def download_dataset():
    """
    Download the Telco Churn dataset.
    Using a publicly available version from IBM Sample Data.
    """
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    print("Downloading Telco Churn dataset...")
    df = pd.read_csv(url)
    
    # Save locally
    df.to_csv('data/telco_churn.csv', index=False)
    print(f"Dataset downloaded successfully! Shape: {df.shape}")
    
    return df

def explore_dataset(df):
    """
    Perform exploratory data analysis on the dataset.
    """
    print("\n" + "="*80)
    print("DATASET OVERVIEW")
    print("="*80)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"Number of Rows: {df.shape[0]}")
    print(f"Number of Columns: {df.shape[1]}")
    
    print("\n" + "-"*80)
    print("COLUMN INFORMATION")
    print("-"*80)
    print(df.info())
    
    print("\n" + "-"*80)
    print("FIRST FEW ROWS")
    print("-"*80)
    print(df.head())
    
    print("\n" + "-"*80)
    print("STATISTICAL SUMMARY")
    print("-"*80)
    print(df.describe())
    
    print("\n" + "-"*80)
    print("MISSING VALUES")
    print("-"*80)
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values found!")
    
    print("\n" + "-"*80)
    print("TARGET VARIABLE DISTRIBUTION (Churn)")
    print("-"*80)
    if 'Churn' in df.columns:
        print(df['Churn'].value_counts())
        print(f"\nChurn Rate: {(df['Churn'] == 'Yes').mean() * 100:.2f}%")
    
    print("\n" + "-"*80)
    print("DATA TYPES")
    print("-"*80)
    print(df.dtypes.value_counts())
    
    return df

def visualize_data(df):
    """
    Create visualizations to understand the data better.
    """
    # Create visualizations directory
    os.makedirs('visualizations', exist_ok=True)
    
    # 1. Target variable distribution
    plt.figure(figsize=(8, 6))
    if 'Churn' in df.columns:
        churn_counts = df['Churn'].value_counts()
        plt.bar(churn_counts.index, churn_counts.values, color=['#2ecc71', '#e74c3c'])
        plt.title('Customer Churn Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Churn Status', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        for i, v in enumerate(churn_counts.values):
            plt.text(i, v + 100, str(v), ha='center', fontweight='bold')
        plt.tight_layout()
        plt.savefig('visualizations/churn_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: churn_distribution.png")
    
    # 2. Numerical features distribution
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numerical_cols:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        for idx, col in enumerate(numerical_cols[:4]):
            axes[idx].hist(df[col].dropna(), bins=30, color='#3498db', edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel(col, fontsize=10)
            axes[idx].set_ylabel('Frequency', fontsize=10)
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/numerical_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: numerical_distributions.png")
    
    # 3. Correlation heatmap for numerical features
    if len(numerical_cols) > 1:
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix - Numerical Features', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: correlation_heatmap.png")
    
    # 4. Categorical features - top categories
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'Churn' in categorical_cols:
        categorical_cols.remove('Churn')
    
    if categorical_cols:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        for idx, col in enumerate(categorical_cols[:4]):
            top_categories = df[col].value_counts().head(10)
            axes[idx].barh(range(len(top_categories)), top_categories.values, color='#9b59b6')
            axes[idx].set_yticks(range(len(top_categories)))
            axes[idx].set_yticklabels(top_categories.index, fontsize=9)
            axes[idx].set_title(f'Top Categories - {col}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Count', fontsize=10)
            axes[idx].grid(alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('visualizations/categorical_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: categorical_distributions.png")
    
    print("\n✓ All visualizations saved to 'visualizations/' directory")

def identify_feature_types(df):
    """
    Identify and categorize features for preprocessing.
    """
    print("\n" + "="*80)
    print("FEATURE CATEGORIZATION")
    print("="*80)
    
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target variable from features
    if 'Churn' in categorical_features:
        categorical_features.remove('Churn')
    
    # Remove customerID if present
    if 'customerID' in categorical_features:
        categorical_features.remove('customerID')
    if 'customerID' in numerical_features:
        numerical_features.remove('customerID')
    
    print(f"\nNumerical Features ({len(numerical_features)}):")
    for feat in numerical_features:
        print(f"  - {feat}")
    
    print(f"\nCategorical Features ({len(categorical_features)}):")
    for feat in categorical_features:
        print(f"  - {feat}")
    
    return numerical_features, categorical_features

def main():
    """
    Main function to orchestrate data exploration.
    """
    print("="*80)
    print("TELCO CUSTOMER CHURN - DATA EXPLORATION")
    print("="*80)
    
    # Download dataset
    df = download_dataset()
    
    # Explore dataset
    df = explore_dataset(df)
    
    # Identify feature types
    numerical_features, categorical_features = identify_feature_types(df)
    
    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    visualize_data(df)
    
    print("\n" + "="*80)
    print("DATA EXPLORATION COMPLETE!")
    print("="*80)
    print("\nNext Steps:")
    print("1. Run 'churn_pipeline.py' to build and train ML models")
    print("2. Review model performance metrics")
    print("3. Export the best model for production use")

if __name__ == "__main__":
    main()
