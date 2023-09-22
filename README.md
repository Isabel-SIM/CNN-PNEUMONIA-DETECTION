# Convolutional Neural Network (CNN) for Pneumonia Detection

This repository contains a Convolutional Neural Network (CNN) implemented using TensorFlow and Keras for the classification of chest X-ray images into two classes: Pneumonia and No Pneumonia. Below is an explanation of the code and the technologies used.

## Convolutional Neural Networks (CNNs)

A Convolutional Neural Network (CNN) is a deep learning architecture widely used for image classification tasks. CNNs are specifically designed to handle grid-like data, such as images, by applying convolutional operations to automatically learn hierarchical features from the input data. These networks consist of convolutional layers, pooling layers, and fully connected layers.

## Technologies Utilised in this Code

### TensorFlow and Keras
- [TensorFlow](https://www.tensorflow.org/) is an open-source deep learning framework developed by Google.
- [Keras](https://keras.io/) is an API that runs on top of TensorFlow, simplifying the process of building and training neural networks.

### ImageDataGenerator
- [`ImageDataGenerator`](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator) is used to perform data augmentation on the training dataset. It rescales pixel values, applies rotations, shifts, shears, zooms, and flips to increase the dataset's diversity and improve model generalization.

### VGG16
- [`VGG16`](https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG16) is a pre-trained convolutional neural network architecture used as the base model for feature extraction. It has been trained on a large dataset and can be fine-tuned for specific tasks.

### Model Architecture
- The model architecture consists of the VGG16 base model, followed by flattening, dense layers with ReLU activation, batch normalization, and dropout layers to prevent overfitting. The final output layer uses sigmoid activation for binary classification.

### Learning Rate Scheduler
- A custom learning rate scheduler is implemented to adjust the learning rate during training. It reduces the learning rate by a factor every few epochs to aid convergence.

### Compilation and Callbacks
- The model is compiled with the Adam optimizer and binary cross-entropy loss. Two callbacks are used: early stopping and reduce learning rate on plateau, to prevent overfitting and improve convergence.

</br>

## Using the Pneumonia Detection App

1. **Upload an X-ray Scan**: To determine if an X-ray scan contains signs of pneumonia, follow these steps:
   - Open the Pneumonia Detection Convolutional Neural Network Streamlit app.
   - Locate the "Upload a chest X-ray image..." section.
   - Click the "Browse" button to select an X-ray scan image from your device. Supported image formats include JPG, JPEG, and PNG.


2. **Image Classification**: Once the image is uploaded, the app will analyse it using the pre-trained Convolutional Neural Network model.
   - The model processes the X-ray image to identify potential signs of pneumonia.
   - Classification results are displayed within moments.
  

3. **View the Prediction**: The app will display the prediction outcome:
   - If the model detects pneumonia-related features, it will indicate "Prediction: Pneumonia."
   - If the model does not detect pneumonia-related features, it will indicate "Prediction: No Pneumonia."


4. **Interpret the Result**: Users can interpret the model's prediction to make informed decisions regarding the presence of pneumonia in the X-ray scan.

<img width="608" alt="Screenshot 2023-09-22 at 3 11 57 pm" src="https://github.com/Isabel-SIM/CONVOLUTIONAL-NEURAL-NETWORK-PNEUMONIA-DETECTION/assets/127584188/4edf58e4-9a87-419d-845e-efda51e72d45">




## Disclaimer

The Pneumonia Detection Convolutional Neural Network and accompanying Streamlit app ("the App") are provided for educational and informational purposes only. While every effort has been made to ensure the accuracy and reliability of the App, it is not a substitute for professional medical advice, diagnosis, or treatment.
