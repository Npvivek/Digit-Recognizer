# Handwritten Digit Recognition using CNN and ANN

## Project Overview

This project uses **Deep Learning** to classify handwritten digits from the **MNIST dataset**. The MNIST dataset contains 28x28 grayscale images of digits from 0 to 9. We explore two different approaches for classification: **Convolutional Neural Networks (CNNs)** and **Artificial Neural Networks (ANNs)**. The goal is to effectively predict the digit in each image and compare the performance of both models.

## Project Structure

- **Convolutional Neural Network (CNN)**: 
  - Built a CNN model with multiple convolutional and max-pooling layers.
  - Used **BatchNormalization** and **Dropout** to enhance generalization.
  - Evaluated using **accuracy**, **confusion matrix**, and **classification report**.

- **Artificial Neural Network (ANN)**:
  - Implemented multiple ANN architectures with different complexities.
  - Compared model performance to determine the most suitable network for digit classification.
  - Evaluated using accuracy, confusion matrix, and classification report.

## Key Features

### Data Preprocessing
- **Normalization**: Pixel values were normalized using **MinMaxScaler** to bring them to the range [0, 1] for both CNN and ANN models.
- **Reshaping**: Reshaped images into tensors for CNN compatibility and into 1D vectors for ANN models.
- **One-hot Encoding**: Converted labels into one-hot encoded format using `to_categorical` to facilitate multi-class classification.

### Exploratory Data Analysis (EDA)
- **Visualization**: Displayed sample images from the dataset to understand the structure and properties.
- **Pixel Range Verification**: Verified pixel values before and after normalization for scaling correctness.

### CNN Model Development
- **Convolutional Layers**: Added multiple layers with **ReLU activation** and **MaxPooling** for feature extraction.
- **Data Augmentation**: Applied **ImageDataGenerator** for data augmentation to enhance generalization by shifting, rotating, and zooming images.
- **Compilation and Optimization**: Used **Adam Optimizer** with a learning rate schedule and **Early Stopping** for better convergence.

### ANN Model Development
- **Multiple Architectures**: Built three different ANN models of varying complexity:
  - **Model_1**: Balanced architecture with moderate complexity.
  - **Model_2**: Simpler architecture with fewer layers and parameters.
  - **Model_3**: Deeper architecture with more layers and higher capacity to handle complexity.
- **Model Selection**: **Model_1** was found to be the most suitable for digit classification, balancing complexity and performance.

### Model Evaluation
- **Metrics**: Used **accuracy**, **F1 score**, **confusion matrix**, and **classification report** for evaluating model performance.
- **Plots**: Displayed training and validation accuracy/loss for all models to visualize learning progress and overfitting behavior.

## Dependencies

- **Programming Language**: Python 3.x
- **Libraries**:
  - Data Manipulation: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Deep Learning: `tensorflow`, `keras`
  - Preprocessing: `scikit-learn`
