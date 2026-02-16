# Customer Churn Prediction ML Pipeline

A production-ready machine learning pipeline for predicting customer churn using the Telco Customer Churn dataset. This project demonstrates end-to-end ML workflow including data preprocessing, model training with hyperparameter tuning, and model deployment.

## 🎯 Project Objectives

- Build reusable ML pipelines using scikit-learn's Pipeline API
- Implement comprehensive data preprocessing (scaling, encoding, imputation)
- Train and compare multiple models (Logistic Regression, Random Forest)
- Perform hyperparameter tuning with GridSearchCV
- Export production-ready models for deployment
- Create prediction scripts for new customer data

## 📊 Dataset

**Telco Customer Churn Dataset**
- Source: IBM Sample Data
- Records: ~7,000 customers
- Features: 20+ (demographic, account, service information)
- Target: Churn (Yes/No)

### Features Include:
- **Demographics**: Gender, SeniorCitizen, Partner, Dependents
- **Account**: Tenure, Contract, PaymentMethod, PaperlessBilling
- **Services**: PhoneService, InternetService, OnlineSecurity, TechSupport, etc.
- **Charges**: MonthlyCharges, TotalCharges

## 🏗️ Project Structure

```
Inter 2/
├── data/                          # Dataset storage
│   └── telco_churn.csv
├── models/                        # Saved pipelines
│   ├── best_churn_pipeline.pkl
│   ├── logistic_regression_pipeline.pkl
│   └── random_forest_pipeline.pkl
├── results/                       # Model performance visualizations
│   ├── model_comparison.png
│   └── confusion_matrices.png
├── visualizations/                # EDA visualizations
│   ├── churn_distribution.png
│   ├── numerical_distributions.png
│   ├── correlation_heatmap.png
│   └── categorical_distributions.png
├── data_exploration.py           # Data download and EDA
├── churn_pipeline.py             # ML pipeline training
├── predict.py                    # Production prediction script
├── requirements.txt              # Project dependencies
└── README.md                     # This file
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- joblib >= 1.3.0

## 📖 Usage

### Step 1: Data Exploration

Download the dataset and perform exploratory data analysis:

```bash
python data_exploration.py
```

**Outputs:**
- Downloads Telco Churn dataset to `data/`
- Displays dataset statistics and information
- Creates visualizations in `visualizations/` directory
- Identifies feature types for preprocessing

### Step 2: Train ML Pipeline

Build, train, and evaluate ML models with hyperparameter tuning:

```bash
python churn_pipeline.py
```

**What it does:**
- Creates preprocessing pipelines for numerical and categorical features
- Trains Logistic Regression with GridSearchCV
- Trains Random Forest with GridSearchCV
- Evaluates both models on test set
- Compares performance metrics
- Saves best model to `models/` directory
- Creates performance visualizations in `results/`

**Hyperparameter Grids:**

*Logistic Regression:*
- C: [0.01, 0.1, 1, 10, 100]
- penalty: ['l2']
- solver: ['lbfgs', 'liblinear']

*Random Forest:*
- n_estimators: [50, 100, 200]
- max_depth: [10, 20, 30, None]
- min_samples_split: [2, 5, 10]
- min_samples_leaf: [1, 2, 4]

### Step 3: Make Predictions

Use the trained pipeline to predict churn for new customers:

```bash
python predict.py
```

**Prediction Methods:**

1. **CSV File Predictions:**
```python
from predict import load_pipeline, predict_from_csv

pipeline = load_pipeline()
results = predict_from_csv(pipeline, 'new_customers.csv')
```

2. **Single Customer Prediction:**
```python
from predict import load_pipeline, predict_from_dict

pipeline = load_pipeline()
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
result = predict_from_dict(pipeline, customer)
```

**Output Format:**
- Prediction: Churn / No Churn
- Churn Probability: 0.0 to 1.0
- Risk Level: Low / Medium / High

## 🔧 Pipeline Architecture

### Preprocessing Pipeline

```python
ColumnTransformer([
    ('numerical', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numerical_features),
    
    ('categorical', Pipeline([
        ('imputer', SimpleImputer(strategy='constant')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]), categorical_features)
])
```

### Complete ML Pipeline

```python
Pipeline([
    ('preprocessor', ColumnTransformer(...)),
    ('classifier', LogisticRegression(...) / RandomForestClassifier(...))
])
```

## 📈 Model Performance

Performance metrics are calculated on the test set (20% of data):

**Evaluation Metrics:**
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix
- Classification Report

**Typical Results:**
- Logistic Regression ROC-AUC: ~0.84
- Random Forest ROC-AUC: ~0.85

## 🎓 Skills Demonstrated

### 1. ML Pipeline Construction
- ✅ Built end-to-end scikit-learn pipelines
- ✅ Integrated preprocessing and modeling steps
- ✅ Created reusable, production-ready code

### 2. Data Preprocessing
- ✅ Handled missing values with SimpleImputer
- ✅ Scaled numerical features with StandardScaler
- ✅ Encoded categorical features with OneHotEncoder
- ✅ Used ColumnTransformer for heterogeneous data

### 3. Hyperparameter Tuning
- ✅ Implemented GridSearchCV for systematic tuning
- ✅ Used cross-validation for robust evaluation
- ✅ Optimized for ROC-AUC metric

### 4. Model Export & Reusability
- ✅ Serialized complete pipelines with joblib
- ✅ Created prediction scripts for new data
- ✅ Ensured production-readiness

### 5. Best Practices
- ✅ Train-test split with stratification
- ✅ Comprehensive evaluation metrics
- ✅ Visualization of results
- ✅ Clean, documented code
- ✅ Modular script organization

## 🚀 Production Deployment

The saved pipeline (`models/best_churn_pipeline.pkl`) is production-ready and can be:

1. **Deployed as REST API** (Flask/FastAPI)
2. **Integrated into web applications**
3. **Used in batch prediction jobs**
4. **Deployed to cloud platforms** (AWS, GCP, Azure)

### Example Flask API:

```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
pipeline = joblib.load('models/best_churn_pipeline.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = pipeline.predict([data])[0]
    probability = pipeline.predict_proba([data])[0, 1]
    
    return jsonify({
        'prediction': 'Churn' if prediction == 1 else 'No Churn',
        'probability': float(probability)
    })

if __name__ == '__main__':
    app.run(debug=True)
```

## 📝 Notes

- The pipeline handles all preprocessing automatically
- No need to manually scale or encode new data
- Missing values are handled by the pipeline
- Unknown categories in new data are handled gracefully

## 🤝 Contributing

This is a portfolio/learning project. Feel free to fork and modify for your own use!

## 📄 License

This project is open source and available for educational purposes.

## 👤 Author

Created as part of a Machine Learning portfolio project demonstrating production-ready ML pipeline development.

---

**Happy Predicting! 🎯**
