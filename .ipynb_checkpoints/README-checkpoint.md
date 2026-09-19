# 🫀 Heart Disease Prediction using KNN

A Machine Learning project that predicts the presence of heart disease based on patient health-related features using the **K-Nearest Neighbors (KNN)** classification algorithm.

The project includes data preprocessing, feature encoding, feature scaling, KNN model training, model evaluation, and a **Streamlit web application** for making predictions.

---

## 📌 Project Overview

Heart disease is influenced by several health-related factors such as age, blood pressure, cholesterol, chest pain, maximum heart rate, and other clinical measurements.

In this project, a KNN classification model is trained on a heart disease dataset and integrated into a Streamlit application.

The application allows users to enter patient information and receive:

- 🧠 Model prediction
- 📊 Heart disease probability
- 📋 Patient input summary
- 🔍 KNN model information

> **Note:** This project is developed for educational and machine learning purposes. It is not intended for medical diagnosis.

---

## 🎯 Objectives

The main objectives of this project are:

1. Understand a real-world classification problem.
2. Perform data preprocessing.
3. Encode categorical features.
4. Scale numerical features.
5. Train a KNN classification model.
6. Evaluate the machine learning model.
7. Save the trained model and preprocessing objects.
8. Build an interactive Streamlit application.
9. Connect the trained ML model with a web interface.

---

## 🤖 Machine Learning Algorithm

### K-Nearest Neighbors (KNN)

KNN is a supervised machine learning algorithm used for classification and regression.

For classification, KNN predicts the class of a new data point based on the classes of its nearest neighbors.

### Basic workflow

```text
New Patient
     │
     ▼
Input Features
     │
     ▼
Data Preprocessing
     │
     ▼
Feature Scaling
     │
     ▼
Find Nearest Neighbors
     │
     ▼
Majority Voting
     │
     ▼
Prediction