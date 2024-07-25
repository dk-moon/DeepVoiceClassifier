# Deep Voice Classifier

## Our Project

### Project Overview

- The proliferation of deep learning and audio synthesis technologies has led to the creation of highly realistic artificial voices, known as deep fake voices. These synthetic voices, which can closely mimic real human speech, pose significant challenges in areas like security, authentication, and media integrity. The need to effectively distinguish between authentic human voices and deep fake voices is more critical than ever.

### Project Objectives

- The primary goal of this project is to develop an accurate and reliable classifier capable of detecting deep fake voices. To achieve this, we will harness advanced machine learning techniques and leverage the capabilities of the Vision Transformer (ViT) model. By converting audio signals into mel-spectrogram images, we can apply sophisticated visual classification methods to identify and differentiate between real and fake voices.

### Key Steps in the Project

1. Library Import:

    - Import necessary libraries such as numpy, pandas, torch, torchaudio, matplotlib, tensorflow, and any others required for processing audio and training the model.

2. Load and Process WAV File:

    - Load your audio data in WAV formar using torchaudio for consistency and preprocesing.

3. Audio Splitting and Noise Reduction:

    - Use an audio separation model to separate human voice from background noise, enhancing the clarity of the audio data.

4. Segment Audio into 1-Second Intervals:

    - Divide the cleaned audio data into 1-second segments for detailed analysis and feature extraction.

5. Convert Segments to Mel-Spectrograms:

    - Convert each 1-second audio segment into a mel-spectrogram using torchaudio.

6. Vision Transformer (ViT) Configuration and Training:

    - Prepare and train the Vision Transformer model using the mel-spectrograms as input data.

7. Model Evaluation:

    - Evaluate the trained model using validation or test data to check its performance.

8. Model Deployment in ONNX Format:

    - Convert the trained model to ONNX format for deployment.

### Inference

 1. Load WAV or MP3 Voice Data:

    - Load the audio data you want to classify using torchaudio. This step ensures that the audio data is in a format suitable for further processing.

 2. Convert Audio Data to Mel-Spectrogram Image:

    - Transform the audio data into a mel-spectrogram image. This involves segmenting the audio if necessary and then applying the mel-spectrogram transformation.

 3. Input Mel-Spectrogram Image to Model:

    - Pass the mel-spectrogram image into the trained Vision Transformer (ViT) model. The mel-spectrograms need to be resized and possibly normalized to match the input requirements of the ViT model.

 4. Predict Class:

    - Use the model to predict the class of each mel-spectrogram segment. The model will output the probability or class indicating whether the voice is a deep fake or a real human voice.

## Introduce DeepVoiceClassifier Flow

### Abstract

The objective of this project is to detect deep fake voices using the FastViT model. By converting audio data into mel-spectrogram images, we can leverage the FastViT model’s capabilities to distinguish between real human voices and deep fake voices. This approach is designed to enhance the accuracy and efficiency of deep fake voice detection.

### Introduction

With the increasing prevalence of deep fake technology, detecting artificially generated voices has become crucial. This project aims to address this challenge by utilizing the FastViT model, a variant of Vision Transformers (ViTs), optimized for speed and performance. By transforming audio signals into mel-spectrogram images, we can effectively apply visual classification techniques to the audio domain.

### Related Work

Previous studies have explored various methods for deep fake voice detection, including traditional machine learning algorithms, convolutional neural networks (CNNs), and recurrent neural networks (RNNs). This project builds upon these foundations by leveraging the advanced capabilities of FastViT, which has shown promise in image classification tasks due to its efficient architecture and high accuracy.

### DeepVoiceClassifier

The core of our approach involves the following steps:

1. Feature Extraction:

    • Audio signals are divided into 1-second segments and converted into mel-spectrogram images.

2. Model Architecture:

    • The FastViT model is employed for its efficiency in handling visual data, adapted here for mel-spectrogram images.

3. Training:

    • The model is trained on a labeled dataset comprising both real human voices and deep fake voices.

4. Inference:

    • The trained model is used to classify new audio samples, providing predictions on whether they are real or fake.

### Experiment

To validate our approach, we conducted experiments using a dataset of labeled audio samples. The dataset includes both genuine human voices and deep fake voices generated using state-of-the-art synthesis techniques. We divided the dataset into training and test sets, trained the FastViT model, and evaluated its performance using standard metrics such as accuracy, precision, recall, and F1-score.

### Conclusion

Our experiments demonstrate that the FastViT model, when applied to mel-spectrogram images, achieves high accuracy in detecting deep fake voices. This approach offers a robust solution for real-time deep fake voice detection, with potential applications in security, authentication, and media verification. Future work may involve further optimization of the model and exploration of additional features to enhance detection capabilities. By leveraging FastViT, a cutting-edge model from Apple’s recent research, we aim to set a new standard in the field of deep fake voice detection.
