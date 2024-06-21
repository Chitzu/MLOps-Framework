# MLOps Framework


## Project Overview 🏥

In this project, I designed and implemented a comprehensive Machine Learning Framework aimed at simplifying the training and testing processes of neural networks. This framework allows users to prepare and preprocess their data before integrating it into a chosen neural network architecture. One of the key features is the ability to adjust hyperparameters through a user-friendly JSON configuration file, which offers flexibility and ease of use. Additionally, the framework logs the results of each training, evaluation, and testing epoch, providing detailed records that facilitate performance monitoring and analysis. This logging functionality ensures that users can track their models progress and make informed adjustments as needed.

## About 

To demonstrate the functionality of this framework, I utilized the Brain Tumor MRI Dataset from Kaggle (available at https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), which is specifically designed for classifying brain tumors. For tackling this multiclass classification problem, I developed a generic Convolutional Neural Network (CNN) architecture composed of two main components.

 - The first component, known as the backbone, consists of 8 Convolutional Layers. These layers are responsible for feature extraction and downsampling, enabling the model to capture relevant features from the input MRI images.
 - The second component, referred to as the neck, includes 3 Fully Connected Layers. These layers are designed for classification purposes, taking the features extracted by the backbone and performing the final classification into the respective tumor categories (glioma, meningioma, notumor, pituitary).
