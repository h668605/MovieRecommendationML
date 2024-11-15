# Movie Recommendation ML Project

## Project Structure

- `notebook.ipynb`: Used to study the dataset and build the model.
- `main.py`: Launches the application and provides movie recommendations to the user.

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/MovieRecommendationML.git
    ```
2. Navigate to the project directory:
    ```bash
    cd MovieRecommendationML
    ```
3. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
4. Ensure you have libraries like scikit-surprise installed:
    ```bash
    pip install scikit-surprise
    ```
5. The project is dependent on some datasets that are too large to upload to github. Go to the link of datasets provided in "Acknowledgements" and download the movies and ratings datasets. Copy these into the CSVFiles folder.


### Usage

1. Run the application:
    ```bash
    python main.py
    ```
2. Press the local link provided in the terminal

3. For each movie you have seen type in the name in the first box. This will filter out titles simular to what you typed so you can easily select your movie. Then you simply give a rating to the movie. Repeat the process for up to five movies.

4. Press the "Get Recommendations"-button. This might take a while so be patient

## Acknowledgements

- Dataset source: https://www.kaggle.com/datasets/samlearner/letterboxd-movie-ratings-data/
