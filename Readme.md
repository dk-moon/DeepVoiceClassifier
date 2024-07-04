# MFCC-FastViT
## Our Project
- Deep Fake Voice Detection

## How to use
### Train
	1.	Load wav or mp3 voice data: Start by loading your audio data in wav or mp3 format.
	2.	Convert audio data to mel-spectrogram image: Transform the audio data into mel-spectrogram images, which represent the audio frequencies over time.
	3.	Prepare X and y data: Use the mel-spectrogram images as the X data (features) and assign the corresponding class (True for real human voice, False for deep fake voice) as the y data (labels).
	4.	Model Training: Train the FastViT model using the prepared X and y data.
	5.	Model Evaluation: Evaluate the trained model using validation or test data to check its performance.

### Inference
	1.	Load wav or mp3 voice data: Load the audio data you want to classify.
	2.	Convert audio data to mel-spectrogram image: Transform the audio data into a mel-spectrogram image.
	3.	Input mel-spectrogram image to model: Pass the mel-spectrogram image into the trained FastViT model.
	4.	Predict class: The model will predict the class, indicating whether the voice is a deep fake or a real human voice.

## Introduce MFCC-FastViT Flow

### Abstract
The objective of this project is to detect deep fake voices using the FastViT model. By converting audio data into mel-spectrogram images, we can leverage the FastViT model’s capabilities to distinguish between real human voices and deep fake voices. This approach is designed to enhance the accuracy and efficiency of deep fake voice detection.

### Introduction
With the increasing prevalence of deep fake technology, detecting artificially generated voices has become crucial. This project aims to address this challenge by utilizing the FastViT model, a variant of Vision Transformers (ViTs), optimized for speed and performance. By transforming audio signals into mel-spectrogram images, we can effectively apply visual classification techniques to the audio domain.

### Related Work
Previous studies have explored various methods for deep fake voice detection, including traditional machine learning algorithms, convolutional neural networks (CNNs), and recurrent neural networks (RNNs). This project builds upon these foundations by leveraging the advanced capabilities of FastViT, which has shown promise in image classification tasks due to its efficient architecture and high accuracy.

### MFCC-FastViT
The core of our approach involves the following steps:

	1.	Feature Extraction: Audio signals are divided into 1-second segments and converted into mel-spectrogram images.
	2.	Model Architecture: The FastViT model is employed for its efficiency in handling visual data, adapted here for mel-spectrogram images.
	3.	Training: The model is trained on a labeled dataset comprising both real human voices and deep fake voices.
	4.	Inference: The trained model is used to classify new audio samples, providing predictions on whether they are real or fake.

### Experiment
To validate our approach, we conducted experiments using a dataset of labeled audio samples. The dataset includes both genuine human voices and deep fake voices generated using state-of-the-art synthesis techniques. We divided the dataset into training and test sets, trained the FastViT model, and evaluated its performance using standard metrics such as accuracy, precision, recall, and F1-score.

### Conclusion
Our experiments demonstrate that the FastViT model, when applied to mel-spectrogram images, achieves high accuracy in detecting deep fake voices. This approach offers a robust solution for real-time deep fake voice detection, with potential applications in security, authentication, and media verification. Future work may involve further optimization of the model and exploration of additional features to enhance detection capabilities. By leveraging FastViT, a cutting-edge model from Apple’s recent research, we aim to set a new standard in the field of deep fake voice detection.