"""
Production-Ready Prediction Script
Load the saved pipeline and make predictions on new customer data.
"""

import pandas as pd
import joblib
import os
import sys

def load_pipeline(model_path='models/best_churn_pipeline.pkl'):
    """
    Load the saved ML pipeline.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    
    print(f"Loading pipeline from {model_path}...")
    pipeline = joblib.load(model_path)
    print("✓ Pipeline loaded successfully!")
    
    return pipeline

def predict_from_csv(pipeline, csv_path):
    """
    Make predictions from a CSV file.
    """
    print(f"\nReading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Remove customerID if present
    customer_ids = None
    if 'customerID' in df.columns:
        customer_ids = df['customerID']
        df = df.drop('customerID', axis=1)
    
    # Remove Churn column if present (for labeled data)
    if 'Churn' in df.columns:
        df = df.drop('Churn', axis=1)
    
    print(f"✓ Loaded {len(df)} records")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = pipeline.predict(df)
    probabilities = pipeline.predict_proba(df)[:, 1]
    
    # Create results dataframe
    results = pd.DataFrame({
        'Prediction': ['Churn' if p == 1 else 'No Churn' for p in predictions],
        'Churn_Probability': probabilities,
        'Risk_Level': ['High' if p > 0.7 else 'Medium' if p > 0.4 else 'Low' for p in probabilities]
    })
    
    if customer_ids is not None:
        results.insert(0, 'customerID', customer_ids)
    
    print("✓ Predictions complete!")
    
    return results

def predict_from_dict(pipeline, customer_data):
    """
    Make prediction from a dictionary of customer features.
    """
    print("\nMaking prediction for single customer...")
    
    # Convert to DataFrame
    df = pd.DataFrame([customer_data])
    
    # Make prediction
    prediction = pipeline.predict(df)[0]
    probability = pipeline.predict_proba(df)[0, 1]
    
    result = {
        'prediction': 'Churn' if prediction == 1 else 'No Churn',
        'churn_probability': probability,
        'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.4 else 'Low'
    }
    
    print("✓ Prediction complete!")
    
    return result

def display_results(results):
    """
    Display prediction results in a formatted way.
    """
    print("\n" + "="*80)
    print("PREDICTION RESULTS")
    print("="*80)
    
    if isinstance(results, pd.DataFrame):
        print(f"\nTotal Predictions: {len(results)}")
        print(f"Predicted Churns: {(results['Prediction'] == 'Churn').sum()}")
        print(f"Predicted No Churns: {(results['Prediction'] == 'No Churn').sum()}")
        print(f"\nRisk Distribution:")
        print(results['Risk_Level'].value_counts().to_string())
        print("\n" + "-"*80)
        print("Sample Predictions:")
        print("-"*80)
        print(results.head(10).to_string(index=False))
    else:
        print(f"\nPrediction: {results['prediction']}")
        print(f"Churn Probability: {results['churn_probability']:.2%}")
        print(f"Risk Level: {results['risk_level']}")

def save_results(results, output_path='predictions.csv'):
    """
    Save prediction results to CSV.
    """
    if isinstance(results, pd.DataFrame):
        results.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to {output_path}")

def main():
    """
    Main function for making predictions.
    """
    print("="*80)
    print("CUSTOMER CHURN PREDICTION")
    print("="*80)
    
    # Load pipeline
    pipeline = load_pipeline()
    
    # Example 1: Predict from CSV file
    print("\n" + "="*80)
    print("EXAMPLE 1: Predict from CSV")
    print("="*80)
    
    # Check if test data exists
    if os.path.exists('data/telco_churn.csv'):
        print("\nUsing sample data from telco_churn.csv...")
        df = pd.read_csv('data/telco_churn.csv')
        
        # Take a sample for demonstration
        sample_df = df.head(20).copy()
        
        # Remove Churn column for prediction
        if 'Churn' in sample_df.columns:
            actual_churn = sample_df['Churn']
            sample_df = sample_df.drop('Churn', axis=1)
        
        # Save sample to temporary file
        sample_df.to_csv('sample_customers.csv', index=False)
        
        # Make predictions
        results = predict_from_csv(pipeline, 'sample_customers.csv')
        
        # Add actual values if available
        if 'actual_churn' in locals():
            results['Actual'] = actual_churn.values
        
        display_results(results)
        save_results(results, 'sample_predictions.csv')
    
    # Example 2: Predict for a single customer
    print("\n" + "="*80)
    print("EXAMPLE 2: Predict for Single Customer")
    print("="*80)
    
    # Example customer data
    customer = {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 12,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 85.0,
        'TotalCharges': 1020.0
    }
    
    print("\nCustomer Profile:")
    for key, value in customer.items():
        print(f"  {key}: {value}")
    
    result = predict_from_dict(pipeline, customer)
    display_results(result)
    
    print("\n" + "="*80)
    print("PREDICTION COMPLETE!")
    print("="*80)
    print("\nUsage Instructions:")
    print("1. For CSV predictions: predict_from_csv(pipeline, 'your_file.csv')")
    print("2. For single customer: predict_from_dict(pipeline, customer_dict)")
    print("3. Results include prediction, probability, and risk level")

if __name__ == "__main__":
    main()
