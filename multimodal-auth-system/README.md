# Multimodal Data Preprocessing Project

## Project Overview
This project focuses on predicting consumer product preferences using multimodal data analysis and machine learning techniques.

## Current Progress

### ✅ Data Collection & Preprocessing
- **Customer Social Profiles**: 155 records with engagement metrics and sentiment analysis
- **Transaction Data**: 150 records with purchase history and ratings
- **Data Quality**: Comprehensive cleaning, duplicate removal, and missing value imputation
- **Feature Engineering**: Customer aggregation and categorical encoding

### ✅ Machine Learning Models
Three predictive models have been implemented and evaluated:

| Model | Accuracy | F1-Score (Weighted) | Status |
|-------|----------|-------------------|---------|
| **XGBoost** | 67.44% | 66.48% | ✅ Best Performer |
| Random Forest | 62.79% | 63.15% | ✅ Implemented |
| LightGBM | 55.81% | 55.09% | ✅ Implemented |

### ✅ Image Processing
- **Facial Image Analysis**: Multi-expression processing (neutral, smiling, surprised)
- **Feature Extraction**: Color histograms, texture analysis, and statistical features
- **Data Augmentation**: Rotation, flip, grayscale, brightness, and blur transformations

## Data Structure
```
data/
├── customer_social_profiles.csv    # Social media engagement data
├── customer_transactions.csv       # Purchase history
├── merged_customer_data.csv        # Aggregated customer profiles
└── merged_customer_data_detailed.csv # Detailed transaction records

media/
├── images/                         # Facial recognition data
│   ├── gustav/                     # Team member images
│   ├── michael/
│   └── reine/
└── audio/                          # Voice recognition data
    ├── gustav/
    └── reine/

models/
├── xgb_model.joblib               # XGBoost model (best performer)
├── rf_model.joblib                # Random Forest model
└── lgbm_model.joblib              # LightGBM model
```

## Key Features
- **Product Category Prediction**: Books, Electronics, Sports, Clothing, Groceries
- **Customer Segmentation**: Based on social media engagement and purchase behavior
- **Multimodal Analysis**: Combining social, transactional, and biometric data

## Technical Stack
- **Data Processing**: Python, Pandas, NumPy
- **Machine Learning**: XGBoost, Random Forest, LightGBM, Scikit-learn
- **Image Processing**: OpenCV, Computer Vision techniques
- **Visualization**: Matplotlib, Seaborn

## Next Steps
- [ ] Audio processing and voice recognition implementation
- [ ] Advanced feature engineering and model optimization
- [ ] Real-time prediction system development
- [ ] Performance monitoring and model deployment
- [ ] Integration of all multimodal components

## Team
Group 17 - Multimodal Data Preprocessing Team

---
*This project is part of ongoing research in multimodal consumer behavior prediction.*