#!/usr/bin/env python3
"""
Multimodal Authentication & Recommendation Simulator
Simulates unauthorized attempts and full transaction flow
"""

import pickle
import joblib
import pandas as pd
import numpy as np
from PIL import Image
import librosa
import argparse
from scipy.spatial.distance import euclidean
import os
import sys
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# Global variables
face_model = None
recommendation_model = None
voice_model = None
voice_scaler = None
voice_features = None
mobilenet = None

def load_models():
    """Load all ML models"""
    global face_model, recommendation_model, voice_model, voice_scaler, voice_features, mobilenet
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(os.path.dirname(script_dir), 'models')
    
    try:
        print("🔄 Loading models...")
        mobilenet = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
        face_model = joblib.load(os.path.join(models_dir, 'facial_recognition_model.pkl'))
        recommendation_model = joblib.load(os.path.join(models_dir, 'xgb_model.joblib'))
        voice_model = joblib.load(os.path.join(models_dir, 'voiceprint_verification_model.pkl'))
        voice_scaler = joblib.load(os.path.join(models_dir, 'voiceprint_scaler.pkl'))
        voice_features = joblib.load(os.path.join(models_dir, 'voiceprint_feature_columns.pkl'))
        print("✅ All models loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        sys.exit(1)

def load_and_preprocess_image(img_path):
    """Load and preprocess image"""
    try:
        img = Image.open(img_path).convert('RGB').resize((224, 224))
        img_array = np.array(img)
        
        img_array = preprocess_input(img_array.astype(np.float32))
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None

def recognize_face(img_path, threshold=0.55, distance_threshold=0.6):
    """Face recognition with distance validation"""
    print(f"🔍 Processing facial recognition: {os.path.basename(img_path)}")
    
    if not os.path.exists(img_path):
        print(f"❌ Image file not found: {img_path}")
        return None
    
    img_tensor = load_and_preprocess_image(img_path)
    if img_tensor is None:
        return None
    
    try:
        # Extract features
        feature_vector = mobilenet.predict(img_tensor, verbose=0)[0]
        feature_vector_reshaped = feature_vector.reshape(1, -1)
        
        # Load known features
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(os.path.dirname(script_dir), 'models')
        with open(os.path.join(models_dir, "known_features.pkl"), "rb") as f:
            known_features, known_labels = pickle.load(f)
        
        # Predict probabilities
        probs = face_model.predict_proba(feature_vector_reshaped)[0]
        predicted_index = np.argmax(probs)
        predicted_class = face_model.classes_[predicted_index]
        confidence = probs[predicted_index]
        
        print(f"   Predicted: {predicted_class} (confidence: {confidence:.2f})")
        
        # Distance comparison
        distances = [euclidean(feature_vector, known_vec) for known_vec in known_features]
        min_distance = min(distances)
        closest_label = known_labels[np.argmin(distances)]
        
        print(f"   Distance to known face: {min_distance:.4f} (closest: {closest_label})")
        
        if confidence >= threshold and min_distance < distance_threshold:
            print(f"✅ Access Granted to: {predicted_class}")
            return predicted_class
        else:
            print(f"❌ Access Denied: Confidence/Distance check failed")
            return None
            
    except Exception as e:
        print(f"❌ Face recognition failed: {e}")
        return None

def generate_recommendations(user_id):
    """Generate product recommendations"""
    print("🎯 Generating product recommendations...")
    
    # Create user-specific features based on user_id
    user_hash = hash(user_id)
    user_features = np.array([[
        user_hash % 1000,           # user_id encoded
        25 + (user_hash % 20),      # age (25-44)
        user_hash % 2,              # gender (0/1)
        40000 + (user_hash % 60000), # income (40k-100k)
        (user_hash % 10) + 1,       # purchase_count (1-10)
        user_hash % 5,              # preferred_category (0-4)
        user_hash % 2,              # is_premium_user (0/1)
        0.5 + (user_hash % 50) / 100, # avg_rating (0.5-1.0)
        50 + (user_hash % 200)      # total_spent (50-250)
    ]])
    
    print(f"   User features: {user_features[0]}")
    recommendations = recommendation_model.predict(user_features)
    print(f"   Model output: {recommendations}")
    print("✅ Recommendations generated (pending voice verification...)")
    return recommendations

def extract_voice_features_dict(audio_file, speaker=None):
    """Extract voice features as dictionary"""
    try:
        y, sr = librosa.load(audio_file)
        duration = librosa.get_duration(y=y, sr=sr)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        mfcc_delta2_mean = np.mean(mfcc_delta2, axis=1)

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(rolloff)
        rolloff_std = np.std(rolloff)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = np.mean(centroid)
        centroid_std = np.std(centroid)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        bandwidth_mean = np.mean(bandwidth)
        bandwidth_std = np.std(bandwidth)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast)
        contrast_std = np.std(contrast)
        flatness = librosa.feature.spectral_flatness(y=y)
        flatness_mean = np.mean(flatness)
        flatness_std = np.std(flatness)

        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_mean = np.mean(f0[~np.isnan(f0)]) if np.any(~np.isnan(f0)) else 0
        f0_std = np.std(f0[~np.isnan(f0)]) if np.any(~np.isnan(f0)) else 0
        f0_min = np.min(f0[~np.isnan(f0)]) if np.any(~np.isnan(f0)) else 0
        f0_max = np.max(f0[~np.isnan(f0)]) if np.any(~np.isnan(f0)) else 0

        order = 12
        autocorr = np.correlate(y, y, mode='full')
        autocorr = autocorr[len(autocorr)//2:len(autocorr)//2+order+1]
        r = autocorr[1:]
        R = autocorr[:-1]
        from scipy.linalg import solve_toeplitz
        lpc_coeffs = solve_toeplitz((R, R), r)[:order]
        lpc_coeffs = np.concatenate(([1], -lpc_coeffs))

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_mean = np.mean(onset_env)
        onset_std = np.std(onset_env)

        features = {
            'file': audio_file,
            'speaker': speaker,
            'duration': duration,
            **{f'mfcc_{i}_mean': mfcc_mean[i] for i in range(13)},
            **{f'mfcc_{i}_std': mfcc_std[i] for i in range(13)},
            **{f'mfcc_delta_{i}_mean': mfcc_delta_mean[i] for i in range(13)},
            **{f'mfcc_delta2_{i}_mean': mfcc_delta2_mean[i] for i in range(13)},
            'rolloff_mean': rolloff_mean,
            'rolloff_std': rolloff_std,
            'centroid_mean': centroid_mean,
            'centroid_std': centroid_std,
            'bandwidth_mean': bandwidth_mean,
            'bandwidth_std': bandwidth_std,
            'contrast_mean': contrast_mean,
            'contrast_std': contrast_std,
            'flatness_mean': flatness_mean,
            'flatness_std': flatness_std,
            'rms_mean': rms_mean,
            'rms_std': rms_std,
            'zcr_mean': zcr_mean,
            'zcr_std': zcr_std,
            'f0_mean': f0_mean,
            'f0_std': f0_std,
            'f0_min': f0_min,
            'f0_max': f0_max,
            **{f'lpc_{i}': lpc_coeffs[i] for i in range(13)},
            **{f'chroma_{i}_mean': chroma_mean[i] for i in range(12)},
            'onset_mean': onset_mean,
            'onset_std': onset_std
        }
        return features
    except Exception as e:
        print(f"❌ Error extracting features: {str(e)}")
        return None

def verify_voice(audio_path, user_id):
    """Voice authentication"""
    print(f"🎤 Processing voice authentication: {os.path.basename(audio_path)}")
    
    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        return False
    
    features_dict = extract_voice_features_dict(audio_path, speaker=user_id)
    if features_dict is None:
        return False
    
    try:
        feature_df = pd.DataFrame([features_dict])
        X_features = feature_df[voice_features].values
        voice_features_scaled = voice_scaler.transform(X_features)
        
        # Get decision function score (distance from boundary)
        decision_score = voice_model.decision_function(voice_features_scaled)[0]
        verification_result = voice_model.predict(voice_features_scaled)[0]
        
        print(f"   Voice model prediction: {verification_result}, score: {decision_score:.4f}")
        
        # Use a more lenient threshold based on decision score
        # Negative scores closer to 0 might still be legitimate
        is_verified = decision_score > -0.5  # Adjust threshold as needed
        
        if is_verified:
            print(f"✅ Voice authentication successful for: {user_id}")
        else:
            print(f"❌ Voice authentication failed for: {user_id}")
        
        return is_verified
    except Exception as e:
        print(f"❌ Voice verification error: {e}")
        return False

def display_recommendations(recommendations, user_id):
    """Display final recommendations"""
    categories = ['Books', 'Electronics', 'Sports', 'Clothing', 'Groceries']
    
    print("\n" + "="*50)
    print("🎉 TRANSACTION APPROVED!")
    print(f"👤 User: {user_id}")
    print("🛍️  PERSONALIZED RECOMMENDATIONS:")
    print("="*50)
    
    for i, rec in enumerate(recommendations, 1):
        if 0 <= rec < len(categories):
            category_name = categories[rec]
        else:
            category_name = f"Category {rec}"
        print(f"   {i}. {category_name}")
    
    print("="*50)

def simulate_unauthorized_attempt():
    """Simulate unauthorized access attempt"""
    print("\n" + "="*60)
    print("🚨 SIMULATING UNAUTHORIZED ACCESS ATTEMPT")
    print("="*60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    media_dir = os.path.join(os.path.dirname(script_dir), 'media')
    
    # Test with unknown face
    unknown_image = os.path.join(media_dir, 'images', 'uknown_face.jpeg')
    unknown_audio = os.path.join(media_dir, 'audio', 'Unknown-voice.m4a')
    
    print("🔍 Step 1: Testing unknown face...")
    user_id = recognize_face(unknown_image)
    
    if user_id is None:
        print("❌ UNAUTHORIZED ACCESS BLOCKED - Unknown face detected")
        return
    
    print("🎯 Step 2: Generating recommendations...")
    recommendations = generate_recommendations(user_id)
    
    print("🎤 Step 3: Testing unknown voice...")
    is_verified = verify_voice(unknown_audio, user_id)
    
    if not is_verified:
        print("❌ UNAUTHORIZED ACCESS BLOCKED - Voice verification failed")
        print("🔒 Recommendations are LOCKED for security")
    else:
        display_recommendations(recommendations, user_id)

def simulate_full_transaction(image_path, audio_path):
    """Simulate complete authorized transaction"""
    print("\n" + "="*60)
    print("✅ SIMULATING AUTHORIZED TRANSACTION")
    print("="*60)
    
    print("🔍 Step 1: Facial Recognition...")
    user_id = recognize_face(image_path)
    
    if user_id is None:
        print("❌ TRANSACTION FAILED - Face recognition denied")
        return False
    
    print("🎯 Step 2: Generating Recommendations...")
    recommendations = generate_recommendations(user_id)
    
    print("🎤 Step 3: Voice Authentication...")
    is_verified = verify_voice(audio_path, user_id)
    
    if is_verified:
        display_recommendations(recommendations, user_id)
        return True
    else:
        print("❌ TRANSACTION FAILED - Voice verification denied")
        print("🔒 Recommendations are LOCKED")
        return False

def interactive_mode():
    """Interactive CLI mode"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    media_dir = os.path.join(os.path.dirname(script_dir), 'media')
    
    print("\n" + "="*60)
    print("🎮 INTERACTIVE MULTIMODAL SIMULATOR")
    print("="*60)
    print("Available test files:")
    print("📸 Images: gustav_neutral.jpeg, michael_smiling.jpeg, reine_neutral.jpeg, uknown_face.jpeg")
    print("🎤 Audio: gustav_confirm.m4a, michael_confirm.m4a, reine_approve.m4a, Unknown-voice.m4a")
    print("="*60)
    
    while True:
        print("\nChoose simulation:")
        print("1. 🚨 Unauthorized attempt (automatic)")
        print("2. ✅ Full transaction (manual input)")
        print("3. 🎯 Quick authorized test")
        print("4. ❌ Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            simulate_unauthorized_attempt()
        
        elif choice == '2':
            image_file = input("Enter image filename: ").strip()
            audio_file = input("Enter audio filename: ").strip()
            
            image_path = os.path.join(media_dir, 'images', image_file)
            audio_path = os.path.join(media_dir, 'audio', audio_file)
            
            simulate_full_transaction(image_path, audio_path)
        
        elif choice == '3':
            # Quick test with known good files
            image_path = os.path.join(media_dir, 'images', 'michael_neutral.jpeg')
            audio_path = os.path.join(media_dir, 'audio', 'michael_confirm.m4a')
            simulate_full_transaction(image_path, audio_path)
        
        elif choice == '4':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please try again.")

def main():
    """Main application"""
    parser = argparse.ArgumentParser(description='Multimodal Authentication Simulator')
    parser.add_argument('--mode', choices=['interactive', 'unauthorized', 'transaction'], 
                       default='interactive', help='Simulation mode')
    parser.add_argument('--image', help='Path to face image file')
    parser.add_argument('--audio', help='Path to voice audio file')
    
    args = parser.parse_args()
    
    print("🚀 MULTIMODAL AUTHENTICATION & RECOMMENDATION SIMULATOR")
    print("="*60)
    
    # Load models
    load_models()
    
    if args.mode == 'unauthorized':
        simulate_unauthorized_attempt()
    
    elif args.mode == 'transaction':
        if not args.image or not args.audio:
            print("❌ Error: --image and --audio required for transaction mode")
            sys.exit(1)
        simulate_full_transaction(args.image, args.audio)
    
    else:
        interactive_mode()

if __name__ == "__main__":
    main()