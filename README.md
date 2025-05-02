import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie dataset
data = {
    'title': ['Inception', 'Titanic', 'The Matrix', 'Interstellar', 'The Notebook'],
    'genres': ['Action Sci-Fi', 'Romance Drama', 'Sci-Fi Action', 'Sci-Fi Drama', 'Romance Drama']
}
movies = pd.DataFrame(data)

# User's preferred genres
user_preferences = "Sci-Fi Action"

# Convert genres to vectors using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(movies['genres'])

# Transform user preference into vector
user_vec = vectorizer.transform([user_preferences])

# Compute cosine similarity
similarity_scores = cosine_similarity(user_vec, tfidf_matrix)

# Recommend top N movies
top_n = 3
top_indices = similarity_scores[0].argsort()[-top_n:][::-1]

# Display recommended movies
print("Recommended Movies:")
for idx in top_indices:
    print(f"- {movies.iloc[idx]['title']}")