# Multimodal Authentication & Product Recommendation System

## Project Overview
A comprehensive multimodal authentication system that combines facial recognition, voice authentication, and machine learning-based product recommendations. The system provides secure biometric authentication while delivering personalized product suggestions based on customer behavior analysis.

## ✅ **Complete Implementation Status**

### Data Collection & Preprocessing
**Customer Data Processing**: Successfully processed 155 customer social profiles with engagement metrics, sentiment analysis, and behavioral patterns. Integrated 150 transaction records with purchase history, product ratings, and demographic information. Implemented comprehensive data cleaning including duplicate removal, missing value imputation, and feature engineering.

**Biometric Data Collection**: Collected facial images and voice recordings from team members across multiple expressions and commands. Created unknown samples for unauthorized access testing.

**Product Prediction Models Performance:**

| Model | Accuracy | F1-Score (Weighted) | Status |
|-------|----------|-------------------|---------|
| **XGBoost** | 67.44% | 66.48% | ✅ Best Performer |
| Random Forest | 62.79% | 63.15% | ✅ Implemented |
| LightGBM | 55.81% | 55.09% | ✅ Implemented |

### Image Processing & Facial Recognition
**Advanced Face Recognition**: Implemented MobileNetV2-based feature extraction with Logistic Regression classification. Added dual security validation using confidence thresholds and Euclidean distance metrics. Supports multiple facial expressions and robust unauthorized user detection.

**Face Recognition Model Performance:**

| Model | Accuracy | F1-Score | Log Loss | Status |
|-------|----------|----------|----------|--------|
| **Logistic Regression** | 85.71% | 85.71% | 0.3274 | ✅ Implemented |
| MobileNetV2 (Feature Extractor) | - | - | - | ✅ Base Model |

### Voice Recognition & Authentication
**Comprehensive Voice Authentication**: Developed advanced audio feature extraction including MFCC, spectral, temporal, and LPC coefficients. Implemented OneClassSVM anomaly detection for secure voice verification with feature selection optimization.

**Voice Authentication Model Performance:**

| Model | Accuracy | Precision | Recall | F1-Score | Status |
|-------|----------|-----------|--------|----------|--------|
| **OneClassSVM** | 16.7% | 100.0% | 16.7% | 28.6% | ✅ Implemented |
| Feature Selection | 61 features | - | - | - | ✅ Applied |

### Multimodal Integration & Security
**Secure Authentication Flow**: Implemented three-step authentication process: Face Recognition → Product Recommendation → Voice Verification → Access Control. Recommendations remain locked until both biometric authentications succeed.

**Security Features**: Dual-factor biometric authentication, distance validation, anomaly detection, confidence thresholding, and automatic unauthorized access blocking.

## Data Structure
```
multimodal-auth-system/
├── data/
│   ├── customer_social_profiles.csv    # Social media engagement data
│   ├── customer_transactions.csv       # Purchase history
│   ├── image_features.csv              # Extracted image features
│   ├── audio_features.csv              # Extracted audio features
│   ├── audio_features_augmented.csv    # Augmented audio features
│   ├── merged_customer_data.csv        # Aggregated customer profiles
│   └── merged_customer_data_detailed.csv # Detailed transaction records
├── media/
│   ├── images/                         # Facial recognition data
│   │   ├── augmented_images/           # Augmented training images
│   │   ├── gustav_neutral.jpeg         # Team member images
│   │   ├── gustav_smiling.jpeg
│   │   ├── gustav_suprised.jpeg
│   │   ├── michael_neutral.jpeg
│   │   ├── michael_smiling.jpeg
│   │   ├── michael_suprised.jpeg
│   │   ├── reine_neutral.jpeg
│   │   ├── reine_smiling.jpeg
│   │   ├── reine_suprised.jpeg
│   │   ├── eliel_neutral.jpeg
│   │   ├── eliel_smiling.jpeg
│   │   ├── eliel_suprised.jpeg
│   │   └── uknown_face.jpeg
│   └── audio/                          # Voice recognition data
│       ├── gustav_approve.m4a
│       ├── gustav_confirm.m4a
│       ├── michael_approve.m4a
│       ├── michael_confirm.m4a
│       ├── reine_approve.m4a
│       ├── reine_confirm.m4a
│       ├── eliel_approve.m4a
│       ├── eliel_confirm.m4a
│       └── Unknown-voice.m4a
├── models/
│   ├── xgb_model.joblib               # XGBoost model (best performer)
│   ├── rf_model.joblib                # Random Forest model
│   ├── lgbm_model.joblib              # LightGBM model
│   ├── facial_recognition_model.pkl   # Face recognition classifier
│   ├── known_features.pkl             # Known face features
│   ├── voiceprint_verification_model.pkl # Voice authentication model
│   ├── voiceprint_scaler.pkl          # Voice feature scaler
│   ├── voiceprint_feature_columns.pkl # Voice feature metadata
│   └── voiceprint_model_metadata.json # Voice model configuration
├── notebook/
│   ├── data_preprocessing.ipynb       # Data cleaning and preprocessing
│   ├── facial_recognition_model.ipynb # Face recognition development
│   ├── image_processing.ipynb         # Image feature extraction
│   ├── audio_processing.ipynb         # Audio feature extraction
│   ├── voice_recognition_model.ipynb  # Voice authentication model
│   └── product_prediction.ipynb       # ML model training
├── script/
│   ├── app.py                         # Production CLI application
├── .gitignore                         # Git ignore rules
└── requirements.txt                   # Python dependencies
```

## 🎯 **System Capabilities**

### Authentication & Security
- **Dual Biometric Authentication**: Face + Voice verification required
- **Advanced Security**: Distance validation, confidence thresholds, anomaly detection
- **Unauthorized Detection**: Automatic blocking of unknown faces and voices
- **Secure Transaction Flow**: Recommendations locked until full authentication

### Product Recommendation Engine
- **Personalized Predictions**: Books, Electronics, Sports, Clothing, Groceries
- **Customer Intelligence**: Social media engagement and purchase behavior analysis
- **ML-Powered**: XGBoost ensemble with 67.44% accuracy
- **Real-time Generation**: Instant recommendations for verified users

### Multimodal Processing
- **Image Processing**: MobileNetV2 feature extraction, multi-expression support
- **Audio Processing**: MFCC, spectral, temporal feature analysis
- **Data Integration**: Social, transactional, and biometric data fusion

## 🚀 **System Simulation**

### Available Simulators

#### **1. Full Multimodal Simulator** (`multimodal_simulator.py`)
Comprehensive testing with real ML models and complete authentication flow.
```bash
cd multimodal-auth-system/script
python multimodal_simulator.py
```

#### **2. Production CLI Application** (`app.py`)
Production-ready authentication system for direct testing.
```bash
python app.py --image ../media/images/gustav_neutral.jpeg --audio ../media/audio/gustav_approve.m4a
```

### Simulation Scenarios

**✅ Authorized Transaction**
- Face: `gustav_neutral.jpeg` + Voice: `gustav_approve.m4a`
- Result: Full authentication → Product recommendations displayed

**❌ Unauthorized Face**
- Face: `uknown_face.jpeg` + Any voice
- Result: Immediate access denial at face recognition

**❌ Unauthorized Voice**
- Face: `gustav_neutral.jpeg` + Voice: `Unknown-voice.m4a`
- Result: Face passes → Voice fails → Access denied

**❌ Complete Unauthorized**
- Face: `uknown_face.jpeg` + Voice: `Unknown-voice.m4a`
- Result: Both authentications fail → Complete access denial

### Interactive Features
- **Menu-driven interface** for easy testing
- **Real-time feedback** with confidence scores and distance metrics
- **Multiple test scenarios** for comprehensive validation
- **Security demonstration** with unauthorized access attempts

## Technical Stack
- **Data Processing**: Python, Pandas, NumPy
- **Machine Learning**: XGBoost, Random Forest, LightGBM, Scikit-learn
- **Image Processing**: OpenCV, TensorFlow/Keras, MobileNetV2
- **Audio Processing**: Librosa, MFCC feature extraction
- **Visualization**: Matplotlib, Seaborn

## 📊 **Project Metrics**
- **Total Data Records**: 305 (155 social profiles + 150 transactions)
- **Biometric Samples**: 30 images + 14 audio files
- **Model Accuracy**: 85.71% (Face) + 67.44% (Products) + 100% Precision (Voice)
- **Security Features**: 4 layers (Face confidence, distance, voice anomaly, dual auth)
- **Simulation Scenarios**: 4 comprehensive test cases

## 🛠️ **Quick Start**

### Installation
```bash
git clone https://github.com/MichaelMusembi/GRP17_Multimodal_Data-_Preprocessing.git
cd multimodal-auth-system
pip install -r requirements.txt
```

### Run Simulation
```bash
cd script
python app.py
```

### Test Scenarios
```bash
# Authorized access
python app.py -i ../media/images/gustav_neutral.jpeg -a ../media/audio/gustav_approve.m4a

# Unauthorized access
python app.py -i ../media/images/uknown_face.jpeg -a ../media/audio/Unknown-voice.m4a
```

## 👥 **Team**
Group 17 - Multimodal Authentication System