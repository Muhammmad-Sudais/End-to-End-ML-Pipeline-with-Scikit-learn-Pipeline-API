# Customer Churn Prediction ML Pipeline - Project Breakdown

---

## 📋 Problem Statement & Objective

### Problem Statement
Customer churn (customer attrition) is a critical business challenge for telecommunications companies. When customers discontinue their services, companies lose revenue and incur high costs to acquire new customers to replace them. Identifying customers at risk of churning before they leave allows companies to take proactive retention measures.

### Business Impact
- **Revenue Loss**: Each churned customer represents lost monthly recurring revenue
- **Acquisition Costs**: Acquiring new customers costs 5-25x more than retaining existing ones
- **Competitive Pressure**: High churn rates indicate customer dissatisfaction and competitive vulnerabilities

### Project Objective
Build a production-ready machine learning pipeline that:
1. **Predicts** which customers are likely to churn with high accuracy
2. **Quantifies** churn risk with probability scores (0-100%)
3. **Identifies** key factors contributing to customer churn
4. **Enables** proactive retention strategies through early warning system
5. **Automates** the entire prediction workflow from raw data to actionable insights

### Success Criteria
- Achieve ROC-AUC score > 80% on test data
- Create reusable, production-ready ML pipeline
- Provide interpretable predictions with probability scores
- Deploy as accessible web application for business users

---

## 📊 Dataset Loading & Preprocessing

### Dataset Overview
**Source**: IBM Telco Customer Churn Dataset  
**Records**: 7,043 customers  
**Features**: 21 columns (19 features + customerID + target)  
**Target Variable**: Churn (Yes/No)  
**Class Distribution**: 
- No Churn: 73.46% (5,174 customers)
- Churn: 26.54% (1,869 customers)

### Feature Categories

#### Numerical Features (3)
- `SeniorCitizen`: Whether customer is 65 or older (0/1)
- `tenure`: Number of months customer has stayed with company (0-72)
- `MonthlyCharges`: Current monthly charge amount ($18.25 - $118.75)

#### Categorical Features (16)
**Demographics:**
- `gender`: Male/Female
- `Partner`: Has partner (Yes/No)
- `Dependents`: Has dependents (Yes/No)

**Services:**
- `PhoneService`: Has phone service (Yes/No)
- `MultipleLines`: Has multiple phone lines (Yes/No/No phone service)
- `InternetService`: Internet service type (DSL/Fiber optic/No)
- `OnlineSecurity`: Has online security add-on (Yes/No/No internet)
- `OnlineBackup`: Has online backup service (Yes/No/No internet)
- `DeviceProtection`: Has device protection plan (Yes/No/No internet)
- `TechSupport`: Has tech support service (Yes/No/No internet)
- `StreamingTV`: Has streaming TV service (Yes/No/No internet)
- `StreamingMovies`: Has streaming movies service (Yes/No/No internet)

**Account Information:**
- `Contract`: Contract type (Month-to-month/One year/Two year)
- `PaperlessBilling`: Uses paperless billing (Yes/No)
- `PaymentMethod`: Payment method (Electronic check/Mailed check/Bank transfer/Credit card)

### Data Quality Assessment
- **Missing Values**: None detected in categorical features
- **Data Type Issues**: `TotalCharges` had some whitespace entries requiring cleaning
- **Class Imbalance**: Moderate (26.54% churn rate) - handled via stratified sampling

### Preprocessing Pipeline Architecture

#### For Numerical Features:
```python
Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
    ('scaler', StandardScaler())                     # Normalize to mean=0, std=1
])
```

#### For Categorical Features:
```python
Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
])
```

#### Combined Preprocessing:
```python
ColumnTransformer([
    ('numerical', numerical_pipeline, numerical_features),
    ('categorical', categorical_pipeline, categorical_features)
])
```

### Data Splitting Strategy
- **Train Set**: 80% (5,634 customers)
- **Test Set**: 20% (1,409 customers)
- **Stratification**: Yes (maintains 26.54% churn rate in both sets)
- **Random State**: 42 (for reproducibility)

---

## 🤖 Model Development & Training

### Model Selection Rationale

#### Model 1: Logistic Regression
**Why chosen:**
- Fast training and prediction
- Provides interpretable coefficients
- Works well for binary classification
- Baseline model for comparison

**Hyperparameter Grid:**
- `C`: [0.01, 0.1, 1, 10, 100] - Regularization strength
- `penalty`: ['l2'] - Ridge regularization
- `solver`: ['lbfgs', 'liblinear'] - Optimization algorithms

**Total Combinations**: 10 parameter sets × 5 CV folds = **50 model fits**

#### Model 2: Random Forest Classifier
**Why chosen:**
- Handles non-linear relationships
- Robust to outliers
- Provides feature importance
- Often superior performance

**Hyperparameter Grid:**
- `n_estimators`: [50, 100, 200] - Number of trees
- `max_depth`: [10, 20, 30, None] - Maximum tree depth
- `min_samples_split`: [2, 5, 10] - Minimum samples to split node
- `min_samples_leaf`: [1, 2, 4] - Minimum samples in leaf node

**Total Combinations**: 108 parameter sets × 5 CV folds = **540 model fits**

### Training Configuration
- **Cross-Validation**: 5-fold stratified CV
- **Optimization Metric**: ROC-AUC (handles class imbalance better than accuracy)
- **Scoring**: Area Under ROC Curve
- **Parallel Processing**: All available CPU cores

### Best Model Parameters

#### Logistic Regression (Best Model)
```python
{
    'C': 100,              # Low regularization (model complexity allowed)
    'penalty': 'l2',       # Ridge regularization
    'solver': 'lbfgs'      # Limited-memory BFGS optimizer
}
```

#### Random Forest (Best Model) 🏆
```python
{
    'n_estimators': 100,        # 100 decision trees
    'max_depth': 10,            # Moderate depth (prevents overfitting)
    'min_samples_split': 10,    # Conservative splitting
    'min_samples_leaf': 4       # Ensures leaf nodes have substance
}
```

### Pipeline Architecture
```python
Pipeline([
    ('preprocessor', ColumnTransformer([...])),  # Automated preprocessing
    ('classifier', RandomForestClassifier(...))   # Best model
])
```

**Key Advantage**: Entire pipeline saved as single object - preprocessing and prediction in one step!

---

## 📈 Evaluation with Relevant Metrics

### Model Comparison

| Metric | Logistic Regression | Random Forest 🏆 |
|--------|---------------------|------------------|
| **Accuracy** | 80.13% | 80.62% |
| **Precision** | 64.67% | 67.47% |
| **Recall** | 54.81% | 52.14% |
| **F1-Score** | 59.33% | 58.82% |
| **ROC-AUC** | 84.59% | **84.69%** ⭐ |

### Winner: Random Forest
Selected based on highest ROC-AUC score (84.69%)

### Detailed Performance Analysis

#### Random Forest Confusion Matrix
```
                 Predicted
                 No Churn  |  Churn
Actual  ─────────────────────────────
No Churn    941 (TN)  |   94 (FP)
Churn       179 (FN)  |  195 (TP)
```

**Interpretation:**
- **True Negatives (941)**: Correctly identified customers who won't churn
- **True Positives (195)**: Correctly identified customers who will churn
- **False Positives (94)**: Incorrectly flagged loyal customers (9.1% false alarm rate)
- **False Negatives (179)**: Missed customers who actually churned (47.9% miss rate)

### Metric Definitions & Business Context

#### Accuracy (80.62%)
- **Definition**: Overall correctness of predictions
- **Formula**: (TP + TN) / Total
- **Business Impact**: 8 out of 10 predictions are correct

#### Precision (67.47%)
- **Definition**: Of customers predicted to churn, how many actually did
- **Formula**: TP / (TP + FP)
- **Business Impact**: When model flags a customer, there's 67% chance they'll actually churn
- **Use Case**: Determines efficiency of retention campaigns (avoid wasting resources)

#### Recall (52.14%)
- **Definition**: Of customers who actually churned, how many did we catch
- **Formula**: TP / (TP + FN)
- **Business Impact**: Model catches 52% of churning customers
- **Trade-off**: Missing 48% of churners, but with lower false alarms

#### F1-Score (58.82%)
- **Definition**: Harmonic mean of precision and recall
- **Formula**: 2 × (Precision × Recall) / (Precision + Recall)
- **Business Impact**: Balanced measure of model effectiveness

#### ROC-AUC (84.69%) ⭐
- **Definition**: Model's ability to distinguish between classes across all thresholds
- **Range**: 0.5 (random) to 1.0 (perfect)
- **Business Impact**: 84.69% probability model ranks a random churner higher than a random non-churner
- **Why Primary Metric**: Robust to class imbalance, evaluates probability calibration

### Probability Calibration
Model outputs probability scores (0-100%) for risk stratification:

| Probability Range | Risk Level | Count | Action Required |
|------------------|------------|-------|-----------------|
| 70-100% | **High Risk** | ~5% | Immediate intervention |
| 40-70% | **Medium Risk** | ~40% | Proactive engagement |
| 0-40% | **Low Risk** | ~55% | Standard service |

---

## 📊 Visualizations

### 1. Churn Distribution
**File**: `visualizations/churn_distribution.png`

**Shows**: Class imbalance in target variable
- No Churn: 73.46% (5,174 customers)
- Churn: 26.54% (1,869 customers)

**Insight**: Moderate imbalance requiring stratified sampling and ROC-AUC metric

---

### 2. Numerical Feature Distributions
**File**: `visualizations/numerical_distributions.png`

**Shows**: Histograms for tenure, MonthlyCharges, SeniorCitizen

**Key Insights**:
- **Tenure**: Bimodal distribution - many new customers (0-10 months) and long-term customers (60+ months)
- **MonthlyCharges**: Right-skewed - most customers pay $20-80/month
- **SeniorCitizen**: Imbalanced - only ~16% are senior citizens

---

### 3. Correlation Heatmap
**File**: `visualizations/correlation_heatmap.png`

**Shows**: Correlation between numerical features and churn

**Key Correlations with Churn**:
- **Tenure**: -0.35 (negative) - Longer tenure = lower churn
- **MonthlyCharges**: +0.19 (positive) - Higher charges = higher churn
- **TotalCharges**: -0.20 (negative) - Higher total spend = lower churn (loyalty)

---

### 4. Categorical Feature Distributions
**File**: `visualizations/categorical_distributions.png`

**Shows**: Bar charts for key categorical features by churn status

**High-Risk Patterns**:
- **Contract**: Month-to-month contracts have 42% churn rate vs 11% for yearly
- **InternetService**: Fiber optic users churn more (42%) vs DSL (19%)
- **PaymentMethod**: Electronic check users churn most (45%)
- **TechSupport**: Customers without tech support churn more (42% vs 15%)

---

### 5. Model Comparison Chart
**File**: `results/model_comparison.png`

**Shows**: Side-by-side bar chart comparing all metrics

**Visualization**: 
- Random Forest slightly outperforms Logistic Regression across most metrics
- Most notable advantage in Precision (67.47% vs 64.67%)

---

### 6. Confusion Matrices
**File**: `results/confusion_matrices.png`

**Shows**: 2×2 confusion matrices for both models

**Visual Comparison**:
- Random Forest: Better precision (fewer false positives)
- Logistic Regression: Better recall (fewer false negatives)
- Trade-off visualization helps stakeholders understand model behavior

---

## 💡 Final Summary / Insights

### Project Achievements ✅

#### Technical Excellence
1. **End-to-End ML Pipeline**: Complete workflow from raw data to production deployment
2. **Automated Preprocessing**: Self-contained pipeline handles all transformations
3. **Rigorous Hyperparameter Tuning**: 590 total model fits via GridSearchCV
4. **Strong Performance**: 84.69% ROC-AUC on unseen test data
5. **Production Deployment**: Flask web app with REST API for real-time predictions

#### Code Quality
- **Modular Design**: Separate scripts for exploration, training, and prediction
- **Reusability**: Saved pipeline works on any new customer data
- **Documentation**: Comprehensive README and execution results
- **Best Practices**: Stratified splits, cross-validation, proper evaluation

---

### Business Insights 🎯

#### High-Risk Customer Profile
Customers most likely to churn exhibit:
- ✗ Month-to-month contracts (easy to cancel)
- ✗ Electronic check payment (less commitment)
- ✗ No security/backup services (low engagement)
- ✗ High monthly charges without added value
- ✗ Low tenure (new customers, < 12 months)
- ✗ Fiber optic internet (possibly due to higher cost)

#### Low-Risk Customer Profile
Loyal customers typically have:
- ✓ Long-term contracts (1-2 years)
- ✓ Automatic payment methods
- ✓ Multiple services bundled
- ✓ Tech support subscription
- ✓ High tenure (> 24 months)
- ✓ Reasonable monthly charges

---

### Actionable Recommendations 💼

#### Immediate Actions (High-Risk Customers - 70%+ probability)
1. **Personal Outreach**: Dedicated account manager contact
2. **Special Offers**: Aggressive retention discounts (20-30% off)
3. **Service Audit**: Review and resolve any service issues
4. **Contract Incentives**: Offer free months for contract upgrade

#### Proactive Strategies (Medium-Risk Customers - 40-70% probability)
1. **Contract Upgrades**: Offer discounts for switching to annual contracts
2. **Service Bundles**: Promote security/backup service packages
3. **Payment Method**: Incentivize automatic payment enrollment
4. **Engagement Programs**: Loyalty rewards, exclusive features
5. **First-Year Focus**: Extra attention to customers in months 1-12

#### Retention Metrics to Track
- **Churn Rate Reduction**: Target 5-10% decrease in next quarter
- **Intervention Success Rate**: % of flagged customers who stay
- **ROI of Retention**: Cost of retention vs. customer lifetime value
- **Model Drift**: Monitor prediction accuracy over time

---

### Model Limitations & Future Improvements 🔄

#### Current Limitations
1. **Recall Trade-off**: Model misses 48% of churning customers
2. **Feature Engineering**: Limited to provided features
3. **Temporal Dynamics**: No time-series patterns captured
4. **External Factors**: Competitor actions, market conditions not included

#### Recommended Enhancements
1. **Feature Engineering**:
   - Customer service call frequency
   - Payment history patterns
   - Usage trends over time
   - Competitor pricing data

2. **Advanced Models**:
   - XGBoost/LightGBM for better performance
   - Neural networks for complex patterns
   - Ensemble methods combining multiple models

3. **Threshold Optimization**:
   - Adjust probability threshold based on business costs
   - Cost-sensitive learning (false negatives more expensive)

4. **Real-Time Monitoring**:
   - Automated retraining pipeline
   - A/B testing framework
   - Model performance dashboards

5. **Explainability**:
   - SHAP values for individual predictions
   - Feature importance analysis
   - Customer-specific churn drivers

---

### Production Deployment Status 🚀

#### Current Capabilities
✅ **Web Application**: Flask app running at http://localhost:5000  
✅ **Single Predictions**: Real-time customer risk assessment  
✅ **Batch Predictions**: CSV upload for bulk processing  
✅ **Risk Stratification**: Automatic categorization (Low/Medium/High)  
✅ **Recommendations**: Personalized retention strategies  

#### API Endpoints
- `GET /`: Web interface for manual predictions
- `POST /predict`: JSON API for single customer prediction
- `POST /batch-predict`: CSV upload for batch predictions

#### Next Deployment Steps
1. **Cloud Hosting**: Deploy to AWS/GCP/Azure
2. **Database Integration**: Connect to customer database
3. **Monitoring**: Set up logging and performance tracking
4. **Automation**: Schedule daily batch predictions
5. **CRM Integration**: Push predictions to sales/support teams

---

### Key Takeaways 🎓

#### For Data Scientists
- Demonstrates end-to-end ML pipeline development
- Shows proper train/test splitting and cross-validation
- Illustrates hyperparameter tuning best practices
- Highlights importance of choosing right evaluation metrics

#### For Business Stakeholders
- **ROI Potential**: Reducing churn by 5% could save millions in revenue
- **Proactive Strategy**: Identify at-risk customers before they leave
- **Resource Optimization**: Focus retention efforts on high-risk customers
- **Data-Driven Decisions**: Replace intuition with predictive insights

#### For Technical Recruiters
- **Production-Ready Code**: Not just a notebook, but deployable system
- **Best Practices**: Proper ML workflow, documentation, testing
- **Full Stack**: Python, scikit-learn, Flask, data visualization
- **Business Acumen**: Translates technical metrics to business value

---

### Project Statistics 📊

| Metric | Value |
|--------|-------|
| **Lines of Code** | ~800 (across 3 main scripts) |
| **Total Model Fits** | 590 (GridSearchCV) |
| **Training Time** | ~5-10 minutes |
| **Model Size** | 4.2 MB (saved pipeline) |
| **Prediction Speed** | < 100ms per customer |
| **Accuracy** | 80.62% |
| **ROC-AUC** | 84.69% |
| **Customers Analyzed** | 7,043 |
| **Features Engineered** | 19 input features |

---

**Project Status**: ✅ **Production Ready**  
**Best Model**: Random Forest Classifier (84.69% ROC-AUC)  
**Deployment**: Flask Web Application  
**Business Impact**: High - Enables proactive customer retention
