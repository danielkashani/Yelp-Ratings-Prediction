
---

# Yelp Restaurant Ratings Prediction: Project Overview

## 1. Introduction

This project analyzes Yelp's business, user, and review datasets to predict restaurant ratings using advanced feature engineering and machine learning models. We focus on extracting, cleaning, and exploring the data, engineering robust features, and benchmarking models including XGBoost, LightGBM, and a simple deep neural network.

---

## 2. Data Extraction and Preprocessing

### 2.1 Extracting and Converting Yelp Datasets

- **Source:** Yelp dataset in `.tar` format (multi-GB scale, exceeds 30GB in memory when uncompressed JSON).
- **Process:**
  1. Extract `.tar` archive.
  2. Convert each JSON file to CSV in manageable chunks to avoid memory crashes.

  ```python
  # Extraction example
  # with tarfile.open('/content/drive/MyDrive/yelp_dataset.tar', "r") as tar:
  #     tar.extractall()
  
  # JSON to CSV conversion in chunks
  chunk_size = 10000
  for chunk in pd.read_json(json_file_path, lines=True, chunksize=chunk_size):
      chunk.to_csv(csv_file_path, mode='a', header=not pd.io.common.file_exists(csv_file_path), index=False)
  ```

- **Storage:** Upload all CSVs to Google Drive for persistent use.

---

## 3. Domain Research: Yelp Dataset Structure

- **Business:** Location, stars, review count, categories, etc.
- **User:** Reviews written, votes (useful/funny/cool), elite status, fans.
- **Review:** Text, rating, votes, business/user linkage.

Our analysis focuses on restaurant performance, user engagement, and potential signs of fake reviews.

---

## 4. Data Loading and Setup

```python
from google.colab import drive
import pandas as pd

drive.mount('/content/drive')
business = pd.read_csv('/content/drive/My Drive/yelp_academic_dataset_business.csv', nrows=500000)
checkin = pd.read_csv('/content/drive/My Drive/yelp_academic_dataset_checkin.csv', nrows=500000)
review = pd.read_csv('/content/drive/My Drive/yelp_academic_dataset_review.csv', nrows=500000)
user = pd.read_csv('/content/drive/My Drive/yelp_academic_dataset_user.csv', nrows=500000)
```

---

## 5. Exploratory Data Analysis (EDA)

- **General Structure:**  
  - Business: 150,346 rows, 14 columns  
  - Review: 500,000 rows, 9 columns  
  - User: 500,000 rows, 22 columns

- **Statistics & Datatypes:**  
  See summary tables in code for detailed info.

- **Missing & Duplicate Values:**  
  Visualized using custom plotting functions.

- **Feature Distributions:**  
  - Histograms and boxplots for stars and review counts.
  - Outlier detection.
  - Correlation heatmaps.

- **Check-in Patterns:**  
  - Check-ins analyzed by day, hour, year.
  - Top businesses by check-in volume identified.

- **Text Analysis:**  
  - Most common words (with/without stopwords).
  - Word clouds for positive/negative reviews.

---

## 6. Data Cleaning & Feature Engineering

- **Column Removal:**  
  Unnecessary columns dropped from each dataset (e.g., postal_code, latitude, review_id, etc.).

- **Feature Engineering:**  
  - `review_length`: Number of words in each review.
  - Dummy variables for state.
  - `elite_count`: Years a user held elite status.
  - `total_hours`: Sum of business opening hours via JSON parsing.

- **Merging Datasets:**  
  - Reviews × Business × User datasets merged for a comprehensive feature set.

- **Temporal Features:**  
  - Years on Yelp.
  - Review year/month/weekday.

---

## 7. Advanced Feature Construction

- **Text Embeddings:**  
  - Word2Vec, GloVe, SentenceTransformer (MiniLM), and BERT (initially tested, but not used in final models).

- **Enhanced Features:**  
  - Word/char counts, sentence counts, exclamation/question counts.
  - Uppercase ratio, average word length.
  - Sentiment polarity and subjectivity.
  - User engagement ratios.

---

## 8. Model Training and Optimization

### 8.1 LightGBM with Optuna

- **Pipeline:**  
  - TF-IDF features (varied n-grams).
  - Numeric and text features combined.
  - Hyperparameter optimization with Optuna (100 trials).

- **Best Parameters:**
  ```
  n_estimators: 235
  learning_rate: 0.0211
  num_leaves: 92
  max_depth: 10
  min_child_samples: 58
  subsample: 0.6341
  colsample_bytree: 0.977
  reg_alpha: 1.75
  reg_lambda: 0.054
  min_split_gain: 0.00158
  ```

- **Performance (Test Set):**
  - **RMSE:** 0.8476
  - **MAE:** 0.6512
  - **R²:** 0.6094

- **Top Feature Importances:**
  | Feature                   | Importance |
  |---------------------------|------------|
  | user_average_star_rating  | 1945       |
  | sentiment_polarity        | 1748       |
  | total_reviews             | 343        |
  | total_user_review_count   | 265        |
  | reviews_per_year          | 263        |
  | ...                       | ...        |

### 8.2 XGBoost and Neural Network

- XGBoost and a simple neural network were benchmarked but did not outperform LightGBM.
- BERT and GloVe embeddings performed poorly for this regression task and were omitted from the final model.

---

## 9. Results & Visualizations

- **Model Comparison:**  
  Final results show LightGBM as the superior model for this dataset and feature set.
- **Feature Importances:**  
  User’s average star rating and sentiment polarity are most predictive.
- **Visualizations:**  
  - Correlation heatmaps
  - Review word clouds
  - Distribution plots for ratings and review counts
  - Statewise restaurant distributions

---

## 10. Usage & Reproducibility

- All code is in Jupyter/Colab notebook format, modularized for each phase (EDA, feature engineering, modeling).
- Data and intermediate files (CSV, embeddings) are stored on Google Drive.

---

## 11. References

- [Yelp Dataset Challenge](https://www.yelp.com/dataset)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/en/latest/)
- [Optuna Documentation](https://optuna.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [TextBlob](https://textblob.readthedocs.io/en/dev/)

---

## 12. Notes

- For large-scale data, always process in chunks to avoid memory issues.
- Feature selection and engineering are critical for performance.
- BERT and GloVe are not always superior for tabular regression tasks.

---
