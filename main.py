# Inez Salazar
# CSCI AI 4710 Final Project
# April 30th 2025
# This project uses csv data from https://www.kaggle.com/datasets/subhajournal/movie-rating
# data includes over 15,000 movies ratings and reviews
# My project extracts word embeddings from a movies description, and uses it to find
# 10 recommended movies for the user based on their input. Also uses genre and rating to narrow down search for users

# libraries : nltk for tokenizing user inputs to words
# gensim to load pre-trained GloVe word embeddings
# numpy for numerical calculations
# punkt for splitting text into words
# glove_model to represent words numerically
# difflib to match genres the user types comparing strings in case of typo

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import gensim.downloader as api
import numpy as np
from numpy.linalg import norm
import difflib

nltk.download('punkt')
glove_model = api.load("glove-wiki-gigaword-50")  # to represent words numerically

# Load CSV file
dataframe = pd.read_csv("data/Rotten Tomatoes Movies.csv")
# Use only these columns from the CSV file
dataframe = dataframe[['movie_title', 'movie_info', 'rating', 'genre']].dropna(subset=['movie_title'])
dataframe = dataframe.fillna("")

# Initialize a graph to show some movie nodes connected to genre and rating nodes
G = nx.Graph()
for _, row in dataframe.iterrows():
    movie_title = row['movie_title']
    rating = row['rating']
    genres = [g.strip() for g in row['genre'].split(',') if g.strip()]

    G.add_node(movie_title, type='movie')
    if rating:
        G.add_node(rating, type='rating')
        G.add_edge(movie_title, rating, relation='has_rating')

    for genre in genres:
        G.add_node(genre, type='genre')
        G.add_edge(movie_title, genre, relation='belongs_to')

# Visualize a small graph with 15 movies
sampled_movies = dataframe['movie_title'].sample(15).tolist()
nodes_to_draw = set(sampled_movies)

for movie in sampled_movies:
    neighbors = list(G.neighbors(movie))
    nodes_to_draw.update(neighbors)

subG = G.subgraph(nodes_to_draw)

node_colors = []
for node in subG.nodes(data=True):
    if node[1]['type'] == 'movie':
        node_colors.append('skyblue')
    elif node[1]['type'] == 'genre':
        node_colors.append('lightgreen')
    elif node[1]['type'] == 'rating':
        node_colors.append('pink')

plt.figure(figsize=(15, 15))
pos = nx.spring_layout(subG)
nx.draw_networkx_nodes(subG, pos, node_color=node_colors, node_size=700)
nx.draw_networkx_edges(subG, pos, edge_color='gray')
nx.draw_networkx_labels(subG, pos, font_size=10, font_weight='bold')
plt.title("Movie Genre Rating Graph", fontsize=20)
plt.axis('off')
plt.show()


movie_info_embeddings = {row['movie_title']: None for _, row in dataframe.iterrows()}

# Function to tokenize text to words, uses words from Glove, returns average mean of text
def get_avg_embedding(text):
    words = word_tokenize(text.lower())
    embeddings = [glove_model[word] for word in words if word in glove_model]
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(50)

# Function to calculate cosine similarity between two vectors
def calculate_similarity(vec1, vec2):
    return np.dot(vec1.flatten(), vec2.flatten()) / (norm(vec1) * norm(vec2) + 1e-8)

# Function to find best matching genre (including partial matches)
def genre_match(movie_genre, selected_genres):
    movie_genres = [g.strip().lower() for g in movie_genre.split(',')]
    for user_genre in selected_genres:
        for movie_genre_item in movie_genres:
            if difflib.SequenceMatcher(None, user_genre, movie_genre_item).ratio() > 0.7:
                return True
    return False

# Function to recommend movies based on selected genres, rating, and user description
def recommend_movies(selected_genres, selected_rating, user_description, top_number=10):
    selected_genres = [genre.strip().lower() for genre in selected_genres]

    filtered_dataframe = dataframe[
        dataframe['genre'].apply(lambda x: genre_match(x, selected_genres)) &
        (dataframe['rating'].str.upper() == selected_rating.upper())
    ]

    if filtered_dataframe.empty:
        return "Sorry, no movies match your genre and rating selection.", None

    user_vec = get_avg_embedding(user_description)
    if np.all(user_vec == 0):
        return "Sorry, I couldn't understand your description.", None

    # Empty dictionary for user description and movie description, iterates over each row of the filtered dataframe
    similarities = {}
    for _, row in filtered_dataframe.iterrows():
        movie_title = row['movie_title']

        if movie_info_embeddings[movie_title] is None:
            movie_info_embeddings[movie_title] = get_avg_embedding(row['movie_info'])

        movie_vec = movie_info_embeddings[movie_title]
        similarity = calculate_similarity(user_vec, movie_vec)
        similarities[movie_title] = similarity

    recommended_movies = sorted(similarities, key=similarities.get, reverse=True)[:top_number] #reverse to sort highest to lowest
    recommendations = [f"{i}. {movie} (similarity: {similarities[movie] * 100:.2f}%)" #two decimal places for %
                       for i, movie in enumerate(recommended_movies, 1)]

    return "\n".join(recommendations), recommended_movies

# main function
def main():
    print("Welcome to the Movie Recommendation System!")
    print("We'll find the best movie for you using preferred genre, rating, and your description.\n")

    while True:
        user_exit = input("Type 'start' to begin or 'bye' to exit: ").strip().lower()

        if user_exit == 'bye':
            print("Chatbot: Goodbye!")
            break
        elif user_exit != 'start':
            print("Chatbot: Invalid input. Please type 'start' or 'bye'.\n")
            continue

        # Pick Genres
        print("\nType a genre from the list (you can choose multiple, separated by commas)")
        available_genres = dataframe['genre'].dropna().unique()
        unique_genres = sorted(set(
            g.strip() for genres in available_genres for g in genres.split(',')
            if g.strip()
        ))

        for idx, genre in enumerate(unique_genres, 1):
            print(f"{idx}. {genre}")

        genre_choice = input("\nEnter your genre or genres here: ").strip()
        if not genre_choice:
            print("Chatbot: You must pick at least one genre. Restarting...\n")
            continue

        selected_genres = [g.strip().lower() for g in genre_choice.split(',')]

        # Pick Rating , use.strip() in case there is empty space
        print("\nChatbot: Pick a Rating")
        possible_ratings = ['G', 'PG', 'PG-13', 'R', 'NC-17', 'NR']
        for rating in possible_ratings:
            print(f"- {rating}")

        rating_choice = input("\nEnter the movie rating you prefer from the list above: ").strip().upper()
        if rating_choice not in possible_ratings:
            print("Chatbot: Invalid rating. Restarting...\n")
            continue

        # Describe Movie
        print("\nStep 3: Describe your ideal movie with as much detail as possible")
        user_description = input("Enter keywords or description with as much detail as possible: ").strip()
        if not user_description:
            print("Chatbot: You must type a description. Restarting...\n")
            continue

        print("\nFinding movies for you...\n")
        recommendations, _ = recommend_movies(selected_genres, rating_choice, user_description)

        print(f"Chatbot: Based on your choices, here are 10 recommendations for you:\n{recommendations}\n")

if __name__ == "__main__":
    main()
