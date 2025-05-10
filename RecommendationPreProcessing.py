# recommendation_model.py

import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("C:/Users/DEEPADHARSHINI/OneDrive/Desktop/Tourism/MergedTourismData.csv")

# Filter necessary columns
df = df[['UserId', 'Attraction', 'Rating']].dropna()

# Encode user and attraction names (optional but helpful)
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

#df['UserId_enc'] = user_encoder.fit_transform(df['UserId'])
df['Attraction_enc'] = item_encoder.fit_transform(df['Attraction'])

# Create user-item matrix (rows: users, columns: attractions, values: ratings)
user_item_matrix = df.pivot_table(index='UserId', columns='Attraction_enc', values='Rating').fillna(0)

# Compute cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix)

# Save everything
joblib.dump(user_item_matrix, 'user_item_matrix.pkl')
joblib.dump(user_similarity, 'user_similarity.pkl')
joblib.dump(user_encoder, 'user_encoder.pkl')
joblib.dump(item_encoder, 'item_encoder.pkl')
print("✅ Model files saved.")
