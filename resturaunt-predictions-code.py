```python
# Data Extraction
# Extracting Yelp's datasets from .tar format Tried to read them as JSON but failed due to memory crashes exceeding 30GB+.
# Extract the .tar file
#with tarfile.open('/content/drive/MyDrive/yelp_dataset.tar', "r") as tar:
#    tar.extractall()
#    tar.list()
# Converted in chunks from JSON to CSV format to avoid memory crashes.
# Convert from JSON to CSV format.
#json_file_path = '/content/yelp_academic_dataset_user.json' # Change depending on what file you want to convert from json to csv
#csv_file_path = '/content/yelp_academic_dataset_user.csv' # Change depending on what file you want to convert from json to csv

# Read the JSON file in chunks
#chunk_size = 10000  # You can adjust this size
#for chunk in pd.read_json(json_file_path, lines=True, chunksize=chunk_size):
#    chunk.to_csv(csv_file_path, mode='a', header=not pd.io.common.file_exists(csv_file_path), index=False)

# Copy over all csv's to google drive for later use.
#!cp /content/yelp_academic_dataset_*.csv /content/drive/My\ Drive/

# Domain Research: Yelp Dataset Overview
# Yelp is a platform for users to review and rate businesses, especially restaurants, to help other users find the best places to eat, drink, and shop.
# This dataset contains various fields:
# Business details: Such as location, rating, category, etc.
# User profiles: Including review counts, useful votes, and badges.
# Reviews: Text of the review, the rating (1-5 stars), and metadata on usefulness, funny, and cool votes.

# For this project, we're analyzing these features to understand restaurant performance, explore patterns in user engagement, and identify potential fake reviews.

# Setup (Library Imports) and Dataset Loading
from google.colab import drive
import tarfile
import os
import geopandas as gpd
import plotly.express as px
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
from gensim.models import Word2Vec
from ipywidgets import interactive, widgets, interact
from IPython.display import display
from wordcloud import WordCloud
import gc
import nltk
from nltk.corpus import stopwords
!pip install transformers torch
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm
nltk.download('stopwords')
STOPWORDS = stopwords.words('english')
# Mount Google Drive
drive.mount('/content/drive')
business = pd.read_csv('/content/drive/My Drive/yelp_academic_dataset_business.csv', nrows=500000)
checkin = pd.read_csv('/content/drive/My Drive/yelp_academic_dataset_checkin.csv', nrows=500000)
review = pd.read_csv('/content/drive/My Drive/yelp_academic_dataset_review.csv', nrows=500000)
user = pd.read_csv('/content/drive/My Drive/yelp_academic_dataset_user.csv', nrows=500000)
# tip = pd.read_csv('/content/drive/My Drive/yelp_academic_dataset_tip.csv')
# tip excluded due to it being a reduced version of review, pointless.
df.info()
# (Pandas df.info() output omitted for brevity)

# Force garbage collection
gc.collect()

# Optionally, you can clear the output of previous cells
# from IPython.display import clear_output
# clear_output()

# Reduce memory usage by downcasting numerical columns to the smallest possible data type
def reduce_mem_usage(df):
  """Iterate through all the columns of a dataframe and modify the data type to reduce memory usage."""
  start_mem = df.memory_usage().sum() / 1024**2
  print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

  for col in df.columns:
    col_type = df[col].dtype

    if col_type != object:
      c_min = df[col].min()
      c_max = df[col].max()
      if str(col_type)[:3] == 'int':
        if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
          df[col] = df[col].astype(np.int8)
        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
          df[col] = df[col].astype(np.int16)
        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
          df[col] = df[col].astype(np.int32)
        elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
          df[col] = df[col].astype(np.int64)
      else:
        if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
          df[col] = df[col].astype(np.float16)
        elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
          df[col] = df[col].astype(np.float32)
        else:
          df[col] = df[col].astype(np.float64)

  end_mem = df.memory_usage().sum() / 1024**2
  print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
  print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

  return df

business = reduce_mem_usage(business)
review = reduce_mem_usage(review)
user = reduce_mem_usage(user)
checkin = reduce_mem_usage(checkin)

# Helper Functions
# Missing Values Graph
def plot_missing_values(df):
    # Get the name of the dataframe
    df_name = [var_name for var_name, var_val in globals().items() if var_val is df][0]

    missing = df.isnull().sum().sort_values(ascending=False)
    missing = missing[missing > 0]

    if missing.empty:
        print(f"No missing values found in the {df_name} dataset.")
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing.index, y=missing.values)
    plt.title(f'Missing Values in {df_name.capitalize()} Dataset', fontsize=22)
    plt.xticks(rotation=90, fontsize=18)
    plt.ylabel('Count', fontsize=18)
    plt.tight_layout()
    plt.show()

# Missing Values per sample Graph
def plot_missing_values_per_sample(df):
    df_name = [var_name for var_name, var_val in globals().items() if var_val is df][0]
    missing_per_sample = df.isnull().sum(axis=1)
    if missing_per_sample.sum() == 0:
        print(f"No missing values in any samples for the {df_name} dataset.")
        return
    plt.figure(figsize=(10, 6))
    sns.histplot(missing_per_sample, bins=30, kde=False)
    plt.title(f'Missing Values per Sample in {df_name.capitalize()} Dataset', fontsize=22)
    plt.xlabel('Number of Missing Values per Sample', fontsize=18)
    plt.ylabel('Count of Samples', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.show()

# Duplicate Values per sample Graph
def plot_duplicate_values_per_sample(df):
    df_name = [var_name for var_name, var_val in globals().items() if var_val is df][0]
    duplicated_rows = df.duplicated(keep=False)
    duplicated_counts = duplicated_rows.value_counts()
    if not duplicated_rows.any():
        print(f"No duplicate rows found in the {df_name} dataset.")
        return
    plt.figure(figsize=(10, 6))
    sns.barplot(x=duplicated_counts.index, y=duplicated_counts.values)
    plt.title(f'Duplicate Values per Sample in {df_name.capitalize()} Dataset', fontsize=22)
    plt.xlabel('Is Duplicate', fontsize=18)
    plt.ylabel('Number of Samples', fontsize=18)
    plt.xticks([0, 1], ['Unique', 'Duplicate'], fontsize=16)
    plt.tight_layout()
    plt.show()

# Function to plot word clouds
def plot_wordcloud(text, title, width=800, height=400, background_color='white'):
    wordcloud = WordCloud(width=width, height=height,
                          background_color=background_color
                          ).generate(' '.join(text))
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=22)
    plt.show()

# Distribution of Number of Reviews Per User
def plot_reviews_per_user(df):
    df_name = [var_name for var_name, var_val in globals().items() if var_val is df][0]
    reviews_per_user = df.groupby('user_id')['total_user_review_count'].sum()
    plt.figure(figsize=(10, 6))
    sns.histplot(reviews_per_user, kde=True, color='skyblue', bins=50)
    plt.title(f'Distribution of Number of Reviews per User in Dataset', fontsize=22)
    plt.xlabel('Number of Reviews per User', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.tight_layout()
    plt.show()

# Relationship Between Review Ratings and Features
def plot_review_ratings_vs_features(df):
    df_name = [var_name for var_name, var_val in globals().items() if var_val is df][0]
    features = ['review_useful', 'review_length', 'elite_count', 'user_useful', 'total_reviews']
    plt.figure(figsize=(18, 12))
    for i, feature in enumerate(features, 1):
        plt.subplot(2, 3, i)
        sns.scatterplot(x=df[feature], y=df['review_star'], color='orange')
        plt.title(f'Review Ratings vs {feature}', fontsize=22)
        plt.xlabel(f'{feature}', fontsize=18)
        plt.ylabel('Review Rating', fontsize=18)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.tight_layout()
    plt.show()

# EDA
# Info & Describe - (Pandas info/describes omitted for brevity)

# Missing/Duplicate Values Analysis
plot_missing_values(business)
plot_missing_values(review)
plot_missing_values(user)
plot_missing_values_per_sample(business)
plot_missing_values_per_sample(review)
plot_missing_values_per_sample(user)
plot_duplicate_values_per_sample(business)
plot_duplicate_values_per_sample(review)
plot_duplicate_values_per_sample(user)

# Features to remove/keep
columns_to_drop_business = ['postal_code', 'latitude', 'longitude', 'is_open',]
business = business.drop(columns=columns_to_drop_business)
columns_to_drop_review = ['review_id', 'funny', 'cool']
review = review.drop(columns=columns_to_drop_review)
columns_to_drop_user = ['funny', 'cool', 'compliment_cute', 'compliment_profile', 'compliment_more',
                        'compliment_funny', 'compliment_cool', 'compliment_note', 'compliment_list',
                        'compliment_hot', 'compliment_plain', 'compliment_writer', 'friends', 'name']
user = user.drop(columns=columns_to_drop_user)

# Initial Feature Engineering and Outliers Detection
review['review_length'] = review['text'].apply(lambda x: len(x.split()))

# Correlation Graphs
plt.figure(figsize=(8, 6))
sns.heatmap(business.corr(numeric_only=True), annot=True, cmap='coolwarm', linewidths=0.5, annot_kws={"size": 16})
plt.title('Correlation Matrix for Business Dataset', fontsize=22)
plt.xlabel('Features', fontsize=18)
plt.ylabel('Features', fontsize=18)
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(review.corr(numeric_only=True), annot_kws={"size": 16}, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix for Review Dataset', fontsize=22)
plt.xlabel('Features', fontsize=18)
plt.ylabel('Features', fontsize=18)
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(user.corr(numeric_only=True), annot=True, annot_kws={"size": 16}, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix for User Dataset', fontsize=22)
plt.xlabel('Features', fontsize=18)
plt.ylabel('Features', fontsize=18)
plt.show()

# Check-in analysis
checkin['date'] = checkin['date'].apply(lambda x: x.split(', '))
checkin = checkin.explode('date')
checkin['date'] = pd.to_datetime(checkin['date'], format='%Y-%m-%d %H:%M:%S')
checkins_per_business = checkin.groupby('business_id').size().reset_index(name='checkin_count')
top_5_checkins = checkins_per_business.sort_values(by='checkin_count', ascending=False).head()
print(top_5_checkins)
checkin['day_of_week'] = checkin['date'].dt.dayofweek
checkins_by_day = checkin.groupby('day_of_week').size()
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
plt.figure(figsize=(10, 6))
sns.barplot(x=days, y=checkins_by_day)
plt.title('Check-ins by Day of the Week', fontsize=22)
plt.xlabel('Day of the Week', fontsize=18)
plt.ylabel('Number of Check-ins', fontsize=18)
plt.show()
checkin['hour'] = checkin['date'].dt.hour
checkins_by_hour = checkin.groupby('hour').size()
plt.figure(figsize=(10, 6))
sns.barplot(x=checkins_by_hour.index, y=checkins_by_hour.values)
plt.title('Check-ins by Hour of the Day', fontsize=22)
plt.xlabel('Hour of the Day', fontsize=18)
plt.ylabel('Number of Check-ins', fontsize=18)
plt.xticks(range(0, 24), fontsize=16)
plt.show()
checkin['year'] = checkin['date'].dt.year
checkins_by_year = checkin.groupby('year').size()
plt.figure(figsize=(10, 6))
sns.lineplot(x=checkins_by_year.index, y=checkins_by_year.values)
plt.title('Check-ins Over Time (Yearly)', fontsize=22)
plt.xlabel('Year', fontsize=18)
plt.ylabel('Number of Check-ins', fontsize=18)
plt.show()

# Feature Distribution
sns.histplot(business['stars'], bins=20)
plt.title('Business Star Rating Distribution', fontsize=22)
plt.xlabel('Stars', fontsize=18)
plt.ylabel('Frequency', fontsize=18)
plt.show()
sns.boxplot(x=business['stars'])
plt.title('Boxplot of Business Star Ratings', fontsize=22)
plt.xlabel('Stars', fontsize=18)
plt.show()
sns.boxplot(x=business['review_count'])
plt.title('Boxplot of Review Counts', fontsize=22)
plt.xlabel('Review Count', fontsize=18)
plt.show()
sns.histplot(business['review_count'], bins=100)
plt.title('Review Count Distribution', fontsize=22)
plt.xlabel('Review Count', fontsize=18)
plt.ylabel('Frequency', fontsize=18)
plt.show()
print(business['review_count'].describe())
sns.histplot(review['stars'], bins=5)
plt.title('Review Star Rating Distribution', fontsize=22)
plt.xlabel('Stars', fontsize=18)
plt.ylabel('Frequency', fontsize=18)
plt.show()

# Multivariate Analysis
sns.scatterplot(x='stars', y='review_count', data=business)
plt.title('Relationship between Stars and Review Count', fontsize=22)
plt.xlabel('Stars', fontsize=18)
plt.ylabel('Review Count', fontsize=18)
plt.show()
sns.scatterplot(y='review_count', x='average_stars', data=user)
plt.title('Relationship between User Review Count and Average Stars', fontsize=22)
plt.xlabel('Average Stars', fontsize=18)
plt.ylabel('Review Count', fontsize=18)
plt.show()
numeric_features = business.select_dtypes(include=[np.number]).columns
numeric_features = [col for col in numeric_features if col != 'stars']
plt.figure(figsize=(15, 5 * ((len(numeric_features) + 1) // 2)))
for i, feature in enumerate(numeric_features, 1):
    plt.subplot(((len(numeric_features) + 1) // 2), 2, i)
    sns.scatterplot(y=feature, x='stars', data=business)
    plt.title(f'{feature} vs stars', fontsize=22)
    plt.xlabel('Stars', fontsize=18)
    plt.ylabel(feature, fontsize=18)
plt.tight_layout()
plt.show()

# Datasets Merge
review_business = pd.merge(review, business, on='business_id', how='left')
combined_data = pd.merge(review_business, user, on='user_id', how='left')
combined_data["total_reviews"] = combined_data["review_count_x"]
combined_data["average_star_rating"] = combined_data["stars_y"]
combined_data["total_user_review_count"] = combined_data["review_count_y"]
combined_data["user_average_star_rating"] = combined_data["average_stars"]
combined_data["review_star"] = combined_data["stars_x"]
combined_data["review_useful"] = combined_data["useful_x"]
combined_data['user_useful'] = combined_data['useful_y']
combined_data["review_date"] = combined_data["date"]
combined_data['elite_count'] = combined_data['elite'].apply(lambda x: len(str(x).split(',')) if isinstance(x, str) else 0)
combined_data.drop(columns=["review_count_x", "stars_x", "stars_y", "review_count_y", "average_stars", "useful_x", "date", "address", 'useful_y', 'elite'], inplace=True)

# Feature Overview and More EDA
numerical_df = combined_data.select_dtypes(include='number')
corr_matrix = numerical_df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
            annot_kws={'size': 16},
            fmt='.2f',
            xticklabels=corr_matrix.columns,
            yticklabels=corr_matrix.columns,
            cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Correlation Matrix Heatmap', fontsize=20)
plt.xticks(fontsize=14, rotation=45)
plt.yticks(fontsize=14, rotation=0)
cbar = plt.gca().collections[0].colorbar
cbar.set_label('Correlation Coefficient', fontsize=14)
plt.tight_layout()
plt.show()

target_variable = 'review_star'
numeric_data = combined_data.select_dtypes(include=[np.number])
correlations = numeric_data.corr()[target_variable].sort_values(ascending=False)
correlations = correlations.drop(target_variable)
plt.figure(figsize=(16, 12))
sns.heatmap(correlations.to_frame(), annot=True, annot_kws={"size": 16}, cmap='coolwarm', vmin=-1, vmax=1)
plt.title(f'Correlation of {target_variable} with Other Variables')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

filtered_words = [word.lower() for text in combined_data['text'] for word in text.split() if word.lower() not in STOPWORDS]
word_counts = Counter(filtered_words)
common_words = word_counts.most_common(10)
plt.figure(figsize=(14, 10))
plt.bar(*zip(*common_words), color='salmon')
plt.title("Top 10 Most Common Words in Reviews (Filtered Stop Words)", fontsize=22)
plt.xlabel("Word", fontsize=18)
plt.ylabel("Frequency", fontsize=18)
plt.xticks(rotation=45, fontsize=14)
plt.yticks(fontsize=14)
plt.show()

fig, axs = plt.subplots(3, 1, figsize=(14, 18))
for i, col in enumerate(['review_useful']):
    if combined_data[combined_data[col] > 0].shape[0] > 0:
        high_rating_reviews = combined_data[combined_data[col] > 0]['text']
        high_rating_words = [word.lower() for text in high_rating_reviews for word in text.split() if word.lower() not in STOPWORDS]
        word_counts = Counter(high_rating_words)
        common_words = word_counts.most_common(10)

        axs[i].bar(*zip(*common_words), color='salmon')
        axs[i].set_title(f"Top 10 Most Common Words in {col.capitalize()} Reviews (Filtered Stop Words)", fontsize=22)
        axs[i].set_xlabel("Word", fontsize=18)
        axs[i].set_ylabel("Frequency", fontsize=18)
        axs[i].tick_params(axis='x', rotation=45, labelsize=14)
        axs[i].tick_params(axis='y', labelsize=14)
    else:
        axs[i].axis('off')
plt.tight_layout()
plt.show()

good_reviews = combined_data[combined_data['review_star'] > 3]['text']
bad_reviews = combined_data[combined_data['review_star'] <= 3]['text']
plot_wordcloud(good_reviews, "Word Cloud for Positive Reviews")
plot_wordcloud(bad_reviews, "Word Cloud for Negative Reviews")

all_words = ' '.join(combined_data['text']).split()
word_counts = Counter(all_words)
common_words = word_counts.most_common(10)
plt.figure(figsize=(10, 6))
plt.bar(*zip(*common_words), color='salmon')
plt.title("Top 10 Most Common Words in Reviews")
plt.xlabel("Word")
plt.ylabel("Frequency")
plt.show()

state_counts = combined_data[combined_data['categories'].str.contains('Restaurants', na=False)].groupby('state').size()
fig = px.choropleth(state_counts.reset_index(), locations='state', locationmode='USA-states', color=0,
                    color_continuous_scale="Blues", scope="usa", title="Number of Restaurants by State")
fig.show()

top_15_users = combined_data.groupby('user_id')['total_user_review_count'].count().nlargest(15)
top_users_data = combined_data[combined_data['user_id'].isin(top_15_users.index)]
top_users_data_clean = top_users_data.dropna(subset=['review_star'])
user_ratings_count = top_users_data_clean.groupby('user_id')['review_star'].nunique()
top_users_data_clean = top_users_data_clean[top_users_data_clean['user_id'].isin(user_ratings_count[user_ratings_count > 1].index)]
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
sns.barplot(x=top_15_users.index, y=top_15_users.values, ax=axes[0], palette='Blues_d')
axes[0].set_title('Top 15 Users by Number of Reviews Written', fontsize=16)
axes[0].set_xlabel('User ID', fontsize=20)
axes[0].set_ylabel('Total Number of Reviews', fontsize=20)
axes[0].tick_params(axis='x', rotation=45, labelsize=12)
axes[0].tick_params(axis='y', labelsize=12)
sns.violinplot(x='user_id', y='review_star', data=top_users_data_clean, ax=axes[1], palette='Blues_d')
axes[1].set_title('Distribution of Ratings Given by Top 15 Users', fontsize=16)
axes[1].set_xlabel('User ID', fontsize=20)
axes[1].set_ylabel('Review Rating (1-5)', fontsize=20)
axes[1].tick_params(axis='x', rotation=45, labelsize=12)
axes[1].tick_params(axis='y', labelsize=12)
axes[1].set_xticks(range(0, len(top_15_users), 2))
plt.tight_layout()
plt.show()

plot_reviews_per_user(combined_data)
plot_review_ratings_vs_features(combined_data)

# Preprocessing Data
df = combined_data.copy()
df.drop(columns=['user_id', 'business_id', 'name', 'city', 'attributes'], inplace=True)
df.dropna(subset=['categories', 'hours'], inplace=True)
df['is_restaurant'] = df['categories'].str.contains('Restaurant', case=False, na=False).astype(int)
df.drop(columns=['categories'], inplace=True)
df = df[df['is_restaurant'] == 1]
def parse_hours(hours):
    if pd.isnull(hours):
        return {}
    try:
        return json.loads(hours.replace("'", "\""))
    except:
        return {}
hours_expanded = df['hours'].apply(parse_hours).apply(pd.Series)
df['total_hours'] = hours_expanded.apply(lambda row: sum(
    [(int(end.split(':')[0]) - int(start.split(':')[0])) for start, end in
     (time_range.split('-') for time_range in row.dropna()) if ':' in start and ':' in end]
), axis=1)
df.drop(columns=['hours'], inplace=True)
df['yelping_since'] = pd.to_datetime(df['yelping_since'])
df['review_date'] = pd.to_datetime(df['review_date'])
df['years_on_yelp'] = (df['review_date'] - df['yelping_since']).dt.days / 365
df.drop(columns=['yelping_since'], inplace=True)
df['review_year'] = df['review_date'].dt.year
df['review_month'] = df['review_date'].dt.month
df['review_weekday'] = df['review_date'].dt.dayofweek  # 0=Monday, 6=Sunday
df.drop(columns=['review_date'], inplace=True)
df = pd.get_dummies(df, columns=['state'], drop_first=True)
dummy_columns = [col for col in df.columns if col.startswith('state_')]
df[dummy_columns] = df[dummy_columns].astype(int)

# Tokenize reviews
tokenized_reviews = df['text'].apply(lambda x: x.split())
w2v_model = Word2Vec(sentences=tokenized_reviews, vector_size=100, window=5, min_count=1, workers=4)
def get_review_embedding(review):
    words = review.split()
    word_vectors = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(100)
df['review_embedding_word2vec'] = df['text'].apply(get_review_embedding)

# Load GloVe embeddings
embeddings_index = {}
with open('/content/drive/My Drive/glove.6B.100d.txt', 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = vector
def get_review_glove_embedding(review):
    words = review.split()
    word_vectors = [embeddings_index[word] for word in words if word in embeddings_index]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(100)
df['review_embedding_glove'] = df['text'].apply(get_review_glove_embedding)

# Load pre-trained SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
df['review_embedding_sentencetransformer'] = list(model.encode(df['text'].tolist(), show_progress_bar=True))

import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

def optimized_bert_embeddings(texts, tokenizer, model, batch_size=64, max_len=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    embeddings = np.zeros((len(texts), model.config.hidden_size))
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting BERT Embeddings"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts.tolist(),
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=max_len
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings[i:i+len(batch_embeddings)] = batch_embeddings
    return embeddings

def extract_bert_embeddings(df, column='text', batch_size=64, max_len=128):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    embeddings = optimized_bert_embeddings(
        df[column],
        tokenizer,
        model,
        batch_size=batch_size,
        max_len=max_len
    )
    return list(embeddings)

df['review_embedding_bert'] = extract_bert_embeddings(df)

import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
df = pd.read_csv("/content/drive/MyDrive/df.csv")
df.to_csv('df.csv', index=False)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
pca = PCA(n_components=2)
glove_2d = pca.fit_transform(list(df['review_embedding_glove']))
sentencetransformer_2d = pca.fit_transform(list(df['review_embedding_sentencetransformer']))
word2vec_2d = pca.fit_transform(list(df['review_embedding_word2vec']))
plt.figure(figsize=(10, 6))
plt.scatter(glove_2d[:, 0], glove_2d[:, 1], alpha=0.5, label="GloVe", color='blue')
plt.title("2D Visualization of GloVe Embeddings (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.show()

# (You can ignore some of this part, since from the README you can tell we used XGBoost, LightGBM and a Simple Neural Network, we discarded BERT and gLovE due to them having very bad performance.)

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

def optimize_xgboost(X_train, y_train, X_test, y_test):
    param_dist = {
        'n_estimators': randint(50, 300),
        'learning_rate': uniform(0.01, 0.3),
        'max_depth': randint(3, 10),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.5, 0.5)
    }
    xgb_model = xgb.XGBRegressor(
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring='neg_mean_absolute_error',
        verbose=1,
        n_jobs=-1
    )
    random_search.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)]
    )
    best_xgb = random_search.best_estimator_
    y_pred_xgb = best_xgb.predict(X_test)
    return {
        'model': best_xgb,
        'best_params': random_search.best_params_,
        'predictions': y_pred_xgb,
        'metrics': {
            'MAE': mean_absolute_error(y_test, y_pred_xgb),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
            'R2': r2_score(y_test, y_pred_xgb)
        }
    }

xgboost_results = optimize_xgboost(X_train, y_train, X_test, y_test)

print("\nXGBoost Results:")
print(f"Best Parameters: {xgboost_results['best_params']}")
for metric, value in xgboost_results['metrics'].items():
    print(f"{metric}: {value}")

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def create_efficient_dl_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            64,
            activation='relu',
            input_dim=input_dim,
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(
            32,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    optimizer = Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999
    )
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    return model

def train_efficient_dl_model(X_train, y_train, X_test, y_test):
    model = create_efficient_dl_model(input_dim=X_train.shape[1])
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    lr_reducer = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001
    )
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping, lr_reducer],
        verbose=0
    )
    y_pred_dl = model.predict(X_test).flatten()
    return {
        'model': model,
        'history': history,
        'predictions': y_pred_dl,
        'metrics': {
            'MAE': mean_absolute_error(y_test, y_pred_dl),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_dl)),
            'R2': r2_score(y_test, y_pred_dl)
        }
    }

def compare_models(X_train, y_train, X_test, y_test):
    xgboost_results = optimize_xgboost(X_train, y_train, X_test, y_test)
    dl_results = train_efficient_dl_model(X_train, y_train, X_test, y_test)
    print("XGBoost Performance:")
    for metric, value in xgboost_results['metrics'].items():
        print(f"{metric}: {value}")
    print("\nDeep Learning Performance:")
    for metric, value in dl_results['metrics'].items():
        print(f"{metric}: {value}")
    return xgboost_results, dl_results

dl_results = train_efficient_dl_model(X_train, y_train, X_test, y_test)

print("\nDeep Learning Results:")
for metric, value in dl_results['metrics'].items():
    print(f"{metric}: {value}")

model_results = {
    'Random Forest': [mae, rmse, r2],
    'XGBoost': [mae_xgb, rmse_xgb, r2_xgb],
    'Deep Learning': [mae_dl, rmse_dl, r2_dl]
}
results_df = pd.DataFrame(model_results, index=['MAE', 'RMSE', 'R²']).T
print(results_df)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([0, 5], [0, 5], color='red', linestyle='--')
plt.title('Random Forest')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred_xgb, alpha=0.6)
plt.plot([0, 5], [0, 5], color='red', linestyle='--')
plt.title('XGBoost')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred_dl, alpha=0.6)
plt.plot([0, 5], [0, 5], color='red', linestyle='--')
plt.title('Deep Learning')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.tight_layout()
plt.show()

def compare_model_performance(rf_results, xgb_results, dl_results):
    print("\n--- Model Performance Comparison ---")
    models = {
        'Random Forest': rf_results,
        'XGBoost': xgb_results,
        'Deep Learning': dl_results
    }
    print("Metric Comparison:")
    for model_name, results in models.items():
        print(f"\n{model_name}:")
        if isinstance(results, dict) and 'metrics' in results:
            for metric, value in results['metrics'].items():
                print(f"{metric}: {value}")
        else:
            print("Results not in expected format")

compare_model_performance(
     {'metrics': {'MAE': rf_mae, 'RMSE': rf_rmse, 'R2': rf_r2}},
     xgboost_results,
     dl_results
 )

# Ignore random forest

import numpy as np
import pandas as pd
!pip install contractions
import time
import scipy.sparse
from scipy.sparse import csr_matrix, hstack
import re
from typing import Dict, Tuple, Any, List
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import contractions
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import optuna
import warnings

warnings.filterwarnings('ignore')

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    numeric_features = [
        'review_length', 'fans', 'total_reviews', 'user_average_star_rating',
        'user_useful', 'total_user_review_count', 'years_on_yelp'
    ]
    state_features = [col for col in df.columns if col.startswith('state_')]
    numeric_features.extend(state_features)
    df['word_count'] = df['text'].str.split().str.len()
    df['char_count'] = df['text'].str.len()
    df['sentence_count'] = df['text'].str.count('[.!?]+')
    df['exclamation_count'] = df['text'].str.count(r'\!')
    df['question_count'] = df['text'].str.count(r'\?')
    df['avg_word_length'] = df['char_count'] / (df['word_count'] + 1)
    df['uppercase_ratio'] = df['text'].apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x))
    )
    sentiments = [TextBlob(str(x)).sentiment for x in df['text']]
    df['sentiment_polarity'] = [s.polarity for s in sentiments]
    df['sentiment_subjectivity'] = [s.subjectivity for s in sentiments]
    if all(col in df.columns for col in ['user_useful', 'total_user_review_count']):
        df['user_engagement_ratio'] = df['user_useful'] / (df['total_user_review_count'] + 1)
    if all(col in df.columns for col in ['years_on_yelp', 'total_user_review_count']):
        df['reviews_per_year'] = df['total_user_review_count'] / (df['years_on_yelp'] + 1)
    new_numeric_features = [
        'word_count', 'char_count', 'avg_word_length', 'sentence_count',
        'exclamation_count', 'question_count', 'uppercase_ratio',
        'sentiment_polarity', 'sentiment_subjectivity'
    ]
    optional_features = ['user_engagement_ratio', 'reviews_per_year']
    new_numeric_features.extend(
        feature for feature in optional_features if feature in df.columns
    )
    numeric_features.extend(new_numeric_features)
    X = df[numeric_features + ['text']].copy()
    X[numeric_features] = X[numeric_features].fillna(X[numeric_features].mean())
    X['text'] = X['text'].fillna('')
    y = df['review_star']
    return X, y

def enhanced_tfidf_features(text_data, max_features=2000):
    tfidf1 = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.9,
        strip_accents='unicode',
        token_pattern=r'\w{1,}'
    )
    tfidf2 = TfidfVectorizer(
        max_features=max_features//2,
        ngram_range=(1, 3),
        min_df=10,
        max_df=0.8,
        strip_accents='unicode',
        token_pattern=r'\w{2,}'
    )
    tfidf_features1 = tfidf1.fit_transform(text_data)
    tfidf_features2 = tfidf2.fit_transform(text_data)
    return hstack([tfidf_features1, tfidf_features2])

def optimize_lgbm(trial, X_train, X_val, y_train, y_val):
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 1e-8, 1.0, log=True)
    }
    model = lgb.LGBMRegressor(**param)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)]
    )
    return np.sqrt(mean_squared_error(y_val, model.predict(X_val)))

def get_feature_importance(model, feature_names: List[str], top_n: int = 20) -> pd.DataFrame:
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    return importance_df.nlargest(top_n, 'importance')

def main(df: pd.DataFrame, n_trials: int = 100):
    print("\nPreparing features...")
    X, y = prepare_features(df)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    print("\nExtracting enhanced text features...")
    text_features_train = enhanced_tfidf_features(X_train['text'])
    text_features_val = enhanced_tfidf_features(X_val['text'])
    text_features_test = enhanced_tfidf_features(X_test['text'])
    numeric_features = [col for col in X.columns if col != 'text']
    scaler = StandardScaler()
    X_train_numeric = scaler.fit_transform(X_train[numeric_features])
    X_val_numeric = scaler.transform(X_val[numeric_features])
    X_test_numeric = scaler.transform(X_test[numeric_features])
    X_train_final = hstack([X_train_numeric, text_features_train])
    X_val_final = hstack([X_val_numeric, text_features_val])
    X_test_final = hstack([X_test_numeric, text_features_test])
    print(f"\nStarting Optuna optimization with {n_trials} trials...")
    start_time = time.time()
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: optimize_lgbm(
        trial, X_train_final, X_val_final, y_train, y_val
    ), n_trials=n_trials)
    print("\nTraining final model with best parameters...")
    best_params = study.best_params
    best_model = lgb.LGBMRegressor(**best_params)
    best_model.fit(X_train_final, y_train)
    test_pred = best_model.predict(X_test_final)
    final_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    final_r2 = r2_score(y_test, test_pred)
    final_mae = mean_absolute_error(y_test, test_pred)
    all_feature_names = numeric_features + [f'tfidf_feature_{i}' for i in range(text_features_train.shape[1])]
    importance_df = get_feature_importance(best_model, all_feature_names)
    print("\n" + "=" * 50)
    print("Final Model Performance")
    print("=" * 50)
    print(f"RMSE: {final_rmse:.4f}")
    print(f"MAE: {final_mae:.4f}")
    print(f"R2 Score: {final_r2:.4f}")
    print("\n" + "=" * 50)
    print("Best Parameters")
    print("=" * 50)
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print("\n" + "=" * 50)
    print("Top Feature Importance")
    print("=" * 50)
    print(importance_df.to_string(index=False))
    total_time = time.time() - start_time
    print(f"\nTotal optimization time: {total_time:.2f} seconds")
    return best_model, final_rmse, final_r2, best_params

if __name__ == "__main__":
    # Load your DataFrame here
    # df = pd.read_csv('your_data.csv')
    best_model, rmse, r2, best_params = main(df, n_trials=100)
    print("\n" + "=" * 50)
    print("Results from Optimization and Model Training")
    print("=" * 50)
    print(f"Final Model Performance on Test Set:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print("\nBest Hyperparameters Found by Optuna:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print("\nFeature Importances (Top 20):")
    print(best_model.feature_importances_)
```
