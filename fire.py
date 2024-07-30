import streamlit as st
import os
import cv2
import numpy as np
import pickle
from skimage.feature import graycomatrix, graycoprops
import tempfile

def get_color_histogram(image):
    histograms = []
    for i in range(3):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        histograms.extend(hist.flatten())
    return histograms

def get_texture_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    asm = graycoprops(glcm, 'ASM')[0, 0]
    return [contrast, dissimilarity, homogeneity, energy, correlation, asm]

def extract_features(image):
    histogram_features = get_color_histogram(image)
    texture_features = get_texture_features(image)
    features = np.array(histogram_features + texture_features).reshape(1, -1)
    return features

def predict_new_image(image, rf_model, svm_model):
    try:
        features = extract_features(image)
        rf_pred = rf_model.predict(features)[0]
        svm_pred = svm_model.predict(features)[0]
        return rf_pred, svm_pred
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

# Load models
try:
    with open('rf_model.pkl', 'rb') as file:
        rf_model = pickle.load(file)
    with open('svm_model.pkl', 'rb') as file:
        svm_model = pickle.load(file)
except Exception as e:
    st.error(f"Error loading models: {e}")

# Streamlit app
st.title('Fire Detection App')

st.write("""
## Fire Detection Project
This project focuses on developing an intelligent fire detection system using machine learning algorithms, including SVM and RF.
""")

option = st.radio('Select Image Source:', ['Upload', 'Camera'])

if option == 'Upload':
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_image:
            temp_image.write(uploaded_file.read())
            temp_image_path = temp_image.name
        image = cv2.imread(temp_image_path)
        os.unlink(temp_image_path)
        if image is not None:
            st.image(image, caption='Uploaded Image', use_column_width=True)
            rf_result, svm_result = predict_new_image(image, rf_model, svm_model)
            if rf_result is not None:
                st.write('### Prediction Results:')
                prediction_data = {
                    'Algorithm': ['Random Forest', 'SVM'],
                    'Prediction': ['FIRE' if rf_result == 1 else 'non-fire',
                                   'FIRE' if svm_result == 1 else 'non-fire']
                }
                st.table(prediction_data)
                contrast, dissimilarity, homogeneity, energy, correlation, asm = get_texture_features(image)
                st.write('### Texture Features:')
                texture_data = {
                    'Feature': ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation', 'ASM'],
                    'Value': [contrast, dissimilarity, homogeneity, energy, correlation, asm]
                }
                st.table(texture_data)
        else:
            st.write("Error reading uploaded image.")

else:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption='Captured Image', use_column_width=True)
            if st.button('Capture'):
                rf_result, svm_result = predict_new_image(frame_rgb, rf_model, svm_model)
                if rf_result is not None:
                    st.write('### Prediction Results:')
                    prediction_data = {
                        'Algorithm': ['Random Forest', 'SVM'],
                        'Prediction': ['FIRE' if rf_result == 1 else 'non-fire',
                                       'FIRE' if svm_result == 1 else 'non-fire']
                    }
                    st.table(prediction_data)
                    cv2.imwrite('captured_image.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    contrast, dissimilarity, homogeneity, energy, correlation, asm = get_texture_features(frame)
                    st.write('### Texture Features:')
                    texture_data = {
                        'Feature': ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation', 'ASM'],
                        'Value': [contrast, dissimilarity, homogeneity, energy, correlation, asm]
                    }
                    st.table(texture_data)
        else:
            st.write("Error capturing image.")
