

import streamlit as st
import librosa
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from PIL import Image
import os

# Define file paths for the trained model and label encoder
MODEL_PATH = r'D:\09_DBDA\09_ML_Project\UrbanSound8K\final_audio_model.keras'
LABELENCODER_PATH = r'D:\09_DBDA\09_ML_Project\UrbanSound8K\final_audio_classes.pkl'
IMAGE_FOLDER = r'D:\09_DBDA\09_ML_Project\UrbanSound8K\images'

# Load the trained deep learning model
model = load_model(MODEL_PATH)

# Load the label encoder to decode predicted classes
with open(LABELENCODER_PATH, 'rb') as file:
    labelencoder = pickle.load(file)

def predict_audio_label(audio_data, sample_rate):
    """
    Predicts the class label for an input audio file.
    
    Parameters:
    audio_data (numpy array): Audio time-series data
    sample_rate (int): Sampling rate of the audio
    
    Returns:
    str: Predicted class label
    """
    # Extract MFCC (Mel-frequency cepstral coefficients) features
    mfccs_features = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features, axis=1)

    # Reshape features to match model input shape
    mfccs_scaled_features = mfccs_scaled_features.reshape(1, 40, 1)

    # Make a prediction
    y_pred = model.predict(mfccs_scaled_features)
    
    # Get the predicted class index
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Decode the predicted class index to its label
    prediction_class = labelencoder.inverse_transform(y_pred_classes)

    return prediction_class[0]

# Streamlit UI setup
st.title("Audio Classification")

# File uploader for audio files
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "mpeg", "ogg"])

if uploaded_file is not None:
    # Load the audio file using librosa
    audio_data, sample_rate = librosa.load(uploaded_file, res_type='kaiser_fast') 

    # Predict the class label
    predicted_label = predict_audio_label(audio_data, sample_rate)

    # Display the predicted label
    st.markdown(f"<h3>Predicted Label: {predicted_label}</h3>", unsafe_allow_html=True)

    # Play the uploaded audio file
    st.audio(uploaded_file, format='audio/wav') 
    
    # Display an image related to the predicted label (if available)
    image_path = os.path.join(IMAGE_FOLDER, f"{predicted_label}.jpg")
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, width=400)
    else:
        st.write("No image found for the predicted label.")
