import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load trained model
model = tf.keras.models.load_model('CHESTXRAY.keras')  

# Define a function to classify the X-ray images
def classify_image(image):
    img = load_img(image, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0  # Normalise
    prediction = model.predict(img)
    return prediction[0][0]

# Streamlit app layout 
st.title("PNEUMONIA DEDECTION")
st.markdown(
    """
    Welcome to the Pneumonia Detection System. This advanced AI-powered application
    utilises state-of-the-art deep learning technology to assist in the analysis of chest X-ray images.
    """
)

uploaded_file = st.file_uploader("Upload a chest X-ray image (JPG, JPEG, or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Chest X-ray", use_column_width=True)
    result = classify_image(uploaded_file)

    prediction_text = "Pneumonia Detected" if result > 0.5 else "No Pneumonia Detected"
    
    # Display classification result
    st.markdown(
        f"""
        ## Analysis Result
        
        The AI-driven analysis indicates the following:
        
        - **Classification**: {prediction_text}
        - **Confidence Score**: {result:.2f}
        
        This information can aid medical professionals in their assessment.
        This application is not a substitute for professional medical advice, diagnosis, or treatment.
        """
    )
