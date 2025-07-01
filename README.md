# Yelp Ratings Prediction

Predicting restaurant ratings from Yelp reviews using machine learning and deep learning models.

## Overview
This project builds and compares multiple regression models to predict restaurant ratings based on 334,000+ Yelp reviews. The goal is to evaluate how different modeling approaches perform on noisy, user-generated text data.

## Features
- NLP preprocessing: text cleaning, tokenization, and TF-IDF
- Feature engineering from unstructured review text
- Comparison between tree-based models (LightGBM, XGBoost) and a neural network
- Evaluation using MAE, MSE, RMSE, and R²

## Results
- Neural network achieved best performance:  
  - R²: 0.74  
  - RMSE: 0.69

## Tools & Technologies
- Python
- Scikit-learn
- TensorFlow
- Pandas, NumPy, Matplotlib
