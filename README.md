# Yelp Ratings Prediction

Predicting restaurant ratings from Yelp reviews using machine learning and deep learning models.

## Overview
This project compares multiple regression models (LightGBM, XGBoost, Neural Network) to predict restaurant ratings from 300k+ Yelp reviews, leveraging advanced feature engineering and NLP.

## Dataset

- [Yelp Open Dataset](https://www.yelp.com/dataset)
- Download and extract the data; convert JSON files to CSV as needed (see [Project.md](./Project.md) for scripts).
- Upload your CSVs to your preferred workspace (Google Drive, local, etc.).

## Installation

```bash
pip install pandas numpy scikit-learn tensorflow lightgbm xgboost optuna nltk textblob wordcloud
```

## Usage

- Open and run `resturaunt-predictions-code.py` in Colab or locally (adjust file paths as needed).
- Example quickstart:
  ```python
  import pandas as pd
  df = pd.read_csv('your_combined_data.csv')
  from resturaunt-predictions-code import main
  best_model, rmse, r2, best_params = main(df, n_trials=100)
  print(f"Test RMSE: {rmse:.4f}, R²: {r2:.4f}")
  ```

## Features

- NLP preprocessing: text cleaning, tokenization, TF-IDF
- Numeric + categorical + text feature engineering
- Model comparison: LightGBM, XGBoost, Neural Network
- Evaluation: MAE, RMSE, R²

## Results

- Best model (LightGBM): RMSE ≈ 0.85, R² ≈ 0.61
- User’s average star rating and sentiment polarity are most predictive features

## Project Structure

- `resturaunt-predictions-code.py` — End-to-end data processing and modeling
- `README.md` — Project summary and quickstart
- `Project.md` — Full technical walkthrough and code snippets

## References

- [Yelp Dataset Challenge](https://www.yelp.com/dataset)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/en/latest/)
- [Optuna](https://optuna.org/)

## License

MIT
