CREATE TABLE users (
    user_id INT PRIMARY KEY,
    gender VARCHAR(1),
    age INT,
    occupation INT,
    zip_code VARCHAR(20)
);

CREATE TABLE movies (
    movie_id INT PRIMARY KEY,
    title TEXT
);

CREATE TABLE genres (
    movie_id INT REFERENCES movies(movie_id),
    genre TEXT,
    PRIMARY KEY (movie_id, genre)
);

CREATE TABLE ratings (
    user_id INT REFERENCES users(user_id),
    movie_id INT REFERENCES movies(movie_id),
    rating FLOAT,
    rated_at TIMESTAMP,
    PRIMARY KEY (user_id, movie_id)
);

CREATE TABLE model_metrics (
    id SERIAL PRIMARY KEY,
    model_name TEXT,
    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    k INT,
    hit_rate FLOAT,
    ndcg FLOAT
);