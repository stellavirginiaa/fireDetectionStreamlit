import streamlit as st
import os
import cv2
import numpy as np
import pickle
from keras.models import load_model
from skimage.feature import graycomatrix, graycoprops
import tempfile

# Function to extract color histogram features from an image
def get_color_histogram(image):
    histograms = []
    for i in range(3):  # For each color channel (B, G, R)
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        histograms.extend(hist.flatten())
    return histograms

# Function to extract texture features using Gray-Level Co-occurrence Matrix (GLCM)
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

# Function to extract features from an image
def extract_features(image):
    histogram_features = get_color_histogram(image)
    texture_features = get_texture_features(image)
    features = np.array(histogram_features + texture_features).reshape(1, -1)
    return features

def predict_new_image(image, rf_model, svm_model, cnn_model):
    features = extract_features(image)

    rf_pred = rf_model.predict(features)[0]
    svm_pred = svm_model.predict(features)[0]

    # Resize image to (128, 128) and add batch dimension
    resized_image = cv2.resize(image, (128, 128))
    resized_image = np.expand_dims(resized_image, axis=0)

    cnn_pred = np.argmax(cnn_model.predict(resized_image / 255.0), axis=1)[0]

    return rf_pred, svm_pred, cnn_pred

# Load the models
with open('rf_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

with open('svm_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)

# Load the CNN model
cnn_model = load_model('cnn_model.h5')

# Streamlit app
st.title('Fire Detection App')
# Narrative
st.write("""
## Fire Detection Project
This project focuses on developing an intelligent fire detection system using machine learning algorithms, including Support Vector Machines (SVM), Random Forest (RF), and Convolutional Neural Networks (CNN). Its primary objective is to provide early detection of fire incidents, minimizing property damage, ensuring public safety, and saving lives.

### Importance and Benefits
Early fire detection is crucial in mitigating property damage, ensuring public safety, and saving lives. Leveraging machine learning algorithms enhances the accuracy and efficiency of fire detection systems, enabling real-time detection and prompt emergency response.

By harnessing the capabilities of SVM, RF, and CNN, the project aims to create a reliable and effective solution for identifying fire occurrences in images or videos, addressing the urgent need for advanced fire detection technologies in various contexts, including wildfire management, industrial safety, and urban fire prevention.
""")

# Option for image source
option = st.radio('Select Image Source:', ['Upload', 'Camera'])

if option == 'Upload':
    # File uploader
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_image:
            temp_image.write(uploaded_file.read())
            temp_image_path = temp_image.name

        # Read the image
        image = cv2.imread(temp_image_path)

        # Remove temporary file
        os.unlink(temp_image_path)

        if image is not None:
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Extract features and make predictions
            rf_result, svm_result, cnn_result = predict_new_image(image, rf_model, svm_model, cnn_model)

            st.write('### Prediction Results:')
            
            # Table for Prediction Results
            prediction_data = {
                'Algorithm': ['Random Forest', 'SVM', 'CNN'],
                'Prediction': [f'FIRE' if rf_result == 1 else 'non-fire',
                               f'FIRE' if svm_result == 1 else 'non-fire',
                               f'FIRE' if cnn_result == 1 else 'non-fire']
            }
            st.table(prediction_data)

            # Display texture features
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
    # Capture image from camera
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Convert the frame from OpenCV BGR format to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the captured image
            st.image(frame_rgb, caption='Captured Image', use_column_width=True)

            # Button to capture image
            if st.button('Capture'):
                # Extract features and make predictions
                rf_result, svm_result, cnn_result = predict_new_image(frame_rgb, rf_model, svm_model, cnn_model)

                st.write('### Prediction Results:')
                
                # Table for Prediction Results
                prediction_data = {
                    'Algorithm': ['Random Forest', 'SVM', 'CNN'],
                    'Prediction': [f'FIRE' if rf_result == 1 else 'non-fire',
                                   f'FIRE' if svm_result == 1 else 'non-fire',
                                   f'FIRE' if cnn_result == 1 else 'non-fire']
                }
                st.table(prediction_data)

                # Save the captured image
                cv2.imwrite('captured_image.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                # Display texture features
                contrast, dissimilarity, homogeneity, energy, correlation, asm = get_texture_features(frame)
                st.write('### Texture Features:')
                texture_data = {
                    'Feature': ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation', 'ASM'],
                    'Value': [contrast, dissimilarity, homogeneity, energy, correlation, asm]
                }
                st.table(texture_data)
