import os

import numpy as np
import pandas as pd


class LoadMovielens1M:
    def __init__(self, path):
        self.path = path

    def _load_ratings(self):
        names = "UserID::MovieID::Rating::Timestamp".split("::")
        ratings = pd.read_csv(
            os.path.join(self.path, "ratings.dat"),
            sep="::",
            names=names,
            engine="python",
        )
        return ratings

    def _load_users(self):
        names = "UserID::Gender::Age::Occupation::Zip-code".split("::")
        users = pd.read_csv(
            os.path.join(self.path, "users.dat"),
            sep="::",
            names=names,
            engine="python",
        )
        return users

    def _load_movies(self):
        names = "MovieID::Title::Genres".split("::")
        movies = pd.read_csv(
            os.path.join(self.path, "movies.dat"),
            sep="::",
            encoding="latin1",
            names=names,
            engine="python",
        )
        return movies

    def load_data(self):
        ratings = self._load_ratings()
        users = self._load_users()
        movies = self._load_movies()
        return (ratings, users, movies)


class PreprocessMovielens1M:
    def __init__(self, datasets):
        self.ratings, self.users, self.movies = datasets

    def _preprocess_ratings(self):
        self.ratings["Rating"] = (self.ratings.Rating > 3).astype(float)

    def _preprocess_users(self):
        self.users["ZipCode"] = pd.factorize(self.users["Zip-code"])[0]
        self.users = self.users.drop("Zip-code", axis=1)
        self.users["Gender"] = pd.factorize(self.users.Gender)[0]
        self.users["Age"] = pd.factorize(self.users.Age)[0]

    def _preprocess_movies(self):

        Genres = [
            "Action",
            "Adventure",
            "Animation",
            "Children's",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "Film-Noir",
            "Horror",
            "Musical",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Thriller",
            "War",
            "Western",
        ]

        self.movies["Year"] = self.movies["Title"].apply(lambda x: x[-5:-1])
        self.movies["Year"] = pd.factorize(self.movies.Year)[0]

        # ジャンルについて onehot化
        for genre in Genres:
            self.movies[genre] = self.movies["Genres"].apply(
                lambda x: genre in x
            )

        # 映画のジャンルについて, bool -> float
        self.movies.iloc[:, 4:] = self.movies.iloc[:, 4:].astype(float)

        # 必要なcolumnのみ抽出
        self.movies = self.movies.loc[:, ["MovieID", "Year"] + Genres]

        # ジャンルを割合で再計算
        self.movies.iloc[:, 2:] = self.movies.iloc[:, 2:].div(
            self.movies.iloc[:, 2:].sum(axis=1), axis=0
        )

    def _incriment_movie_view(self, x):
        tmp = np.zeros((self.movies.MovieID.max() + 1,))
        for MovieID in x.MovieID:
            tmp[MovieID] += 1
        return pd.Series(tmp)

    def _get_rating_history(self):
        return self.ratings.groupby("UserID").apply(self._incriment_movie_view)

    def apply_prepocess(self):
        self._preprocess_ratings()
        self._preprocess_users()
        self._preprocess_movies()
        self._movie_id_remap()
        self._user_id_remap()
        history = self._get_rating_history()
        return (self.ratings, self.users, self.movies, history)

    def _get_movie_id_map(self):
        movie_id_map = {}
        for i in range(self.movies.shape[0]):
            movie_id_map[self.movies.loc[i, "MovieID"]] = i
        return movie_id_map

    def _get_user_id_map(self):
        user_id_map = {}
        for i in range(self.users.shape[0]):
            user_id_map[self.users.loc[i, "UserID"]] = i
        return user_id_map

    def _user_id_remap(self):
        user_id_map = self._get_user_id_map()
        self.users["UserID"] = self.users["UserID"].apply(
            lambda x: user_id_map[x]
        )
        self.ratings["UserID"] = self.ratings["UserID"].apply(
            lambda x: user_id_map[x]
        )

    def _movie_id_remap(self):
        movie_id_map = self._get_movie_id_map()
        self.movies["MovieID"] = self.movies["MovieID"].apply(
            lambda x: movie_id_map[x]
        )
        self.ratings["MovieID"] = self.ratings["MovieID"].apply(
            lambda x: movie_id_map[x]
        )


def get_MovieLens1M(path):
    load_rawML1M = LoadMovielens1M(path)
    raw_datasets = load_rawML1M.load_data()

    load_processedML1M = PreprocessMovielens1M(raw_datasets)
    processed_data = load_processedML1M.apply_prepocess()
    return raw_datasets, processed_data
