import gradio as gr
import pandas as pd
import os
from surprise import SVD, Dataset, Reader

current_dir = os.path.dirname(__file__)

# Load csv-files
file_path_movies = os.path.join(current_dir, "CSVFiles", "movies_data_cleaned.csv")
file_path_ratings = os.path.join(current_dir, "CSVFiles", "ratings_export.csv")

movies = pd.read_csv(file_path_movies)
global ratings
ratings = pd.read_csv(file_path_ratings)
ratings = ratings[['movie_id', 'rating_val', 'user_id']]

# Filter ratings to include only those with movie_ids present in movies DataFrame
ratings = ratings[ratings['movie_id'].isin(movies['movie_id'])]

user_id = 'dummy_user_id'

global algo
algo = SVD()

movie_titles = movies['movie_title'].tolist()  # List of movie titles
movie_ratings = []

def recommend_movies():
    all_movie_ids = ratings['movie_id'].unique()

    # Get the movie IDs the user has already rated
    rated_movie_ids = ratings[ratings['user_id'] == user_id]['movie_id'].unique()

    # Find the movies the user has not rated
    unrated_movie_ids = [movie_id for movie_id in all_movie_ids if movie_id not in rated_movie_ids]

    # Predict ratings for all unrated movies
    predictions = [algo.predict(user_id, movie_id) for movie_id in unrated_movie_ids]

    # Sort predictions by estimated rating in descending order
    sorted_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)

    # Get top N recommendations
    top_n = 5  # Number of recommendations to display
    top_recommendations = [(pred.iid, pred.est) for pred in sorted_predictions[:top_n]]
    return top_recommendations

def train_model():
    global algo
    global ratings
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating_val']], reader)
    trainset = data.build_full_trainset()
    algo.fit(trainset)

def add_movie(*args):
    global ratings
    num_movies = len(args) // 2
    movie_names = args[:num_movies]  # First half of arguments are movie names
    user_ratings = args[num_movies:]  # Second half of arguments are ratings
    movie_ratings.clear()
    new_ratings = []

    for movie_name, rating in zip(movie_names, user_ratings):
        if movie_name:  # Check if movie_name is not empty
            movie_id = movies.loc[movies['movie_title'] == movie_name, 'movie_id'].values[0]
            movie_ratings.append({"Movie ID": movie_id, "Rating": rating})
            new_ratings.append({"movie_id": movie_id, "rating_val": rating, "user_id": user_id})

    # Append new ratings to the ratings DataFrame
    ratings = ratings._append(new_ratings, ignore_index=True)
    return movie_ratings

# Function to filter movies based on search query
def filter_movies(query):
    filtered_movies = movies[movies['movie_title'].str.contains(query, case=False, na=False)]
    return filtered_movies['movie_title'].tolist()

# Gradio interface
with gr.Blocks() as demo:
    movie_inputs = []
    rating_inputs = []

    for i in range(5):  # Assuming you want 5 movie inputs
        search_box = gr.Textbox(label=f"Search for Movie {i+1}")
        movie_input = gr.Dropdown(choices=[], label=f"Movie Name {i+1}")
        search_box.change(lambda query, movie_input=movie_input: gr.update(choices=filter_movies(query)), inputs=search_box, outputs=movie_input)
        rating_input = gr.Slider(minimum=1, maximum=10, step=1, label=f"Rating {i+1}")
        movie_inputs.append(movie_input)
        rating_inputs.append(rating_input)

    # Button to add ratings and display them
    submit_button = gr.Button("Get recommendations")
    recommendation_output = gr.Textbox(label="Recommendations")

    def on_submit(*args):
        movie_ratings = add_movie(*args)
        train_model()
        recommendations = recommend_movies()
        recommendation_text = "Here are some movies we think you would like!\n"
        for movie_id, rating in recommendations:
            movie_title = movies.loc[movies['movie_id'] == movie_id, 'movie_title']
            if not movie_title.empty:
                recommendation_text += f"{movie_title.values[0]}. Your predicted rating: {rating:.2f}\n"
        return recommendation_text

    submit_button.click(
        on_submit,
        inputs=movie_inputs + rating_inputs,  # Flattened list of inputs
        outputs=recommendation_output
    )

demo.launch(share=True)
