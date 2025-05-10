import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved components
user_item_matrix = joblib.load("user_item_matrix.pkl")
user_matrix_reduced = joblib.load("user_matrix_reduced.pkl")
knn_model = joblib.load("knn_model.pkl")

# Define user input interface
def user_input():
    st.title("🎯 Tourism Recommender System")
    st.markdown("Enter a **User ID** to get 5 personalized attraction recommendations.")

    user_id = st.number_input("🔢 Enter User ID", min_value=1, step=1)
    return user_id

# Get user input
user_id_input = user_input()

# Recommendation logic
def recommend_attractions(user_id, user_item_matrix, user_matrix_reduced, knn_model, num_recommendations=5):
    if user_id not in user_item_matrix.index:
        return None

    user_idx = user_item_matrix.index.get_loc(user_id)
    distances, indices = knn_model.kneighbors([user_matrix_reduced[user_idx]], n_neighbors=5)
    similar_users = user_item_matrix.index[indices.flatten()[1:]]

    user_ratings = user_item_matrix.loc[user_id]
    unseen_attractions = user_ratings[user_ratings == 0].index

    attraction_scores = {}
    for sim_user in similar_users:
        for attraction in unseen_attractions:
            attraction_scores[attraction] = attraction_scores.get(attraction, 0) + user_item_matrix.loc[sim_user, attraction]

    recommended = sorted(attraction_scores, key=attraction_scores.get, reverse=True)[:num_recommendations]
    return recommended if recommended else []

# Trigger recommendation
if st.button("🔍 Get Recommendations"):
    recommendations = recommend_attractions(user_id_input, user_item_matrix, user_matrix_reduced, knn_model)
    
    if recommendations:
        st.success("✅ Top 5 Recommended Attractions:")
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
    else:
        st.warning("⚠️ No recommendations found. Try another User ID.")
