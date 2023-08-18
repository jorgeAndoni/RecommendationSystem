import utils
from sklearn.metrics import pairwise_distances
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import configuration as conf

class FBCKnnMethod(object):

    '''
    Metodo Knn e suas tres variacoes (baseado nos dados dos usuarios, generos dos filmes e revisoes dos filmes)
    '''

    def __init__(self, train_rating_matrix, test_data, global_mean, method, k):
        self.train_rating_matrix = train_rating_matrix
        self.test_data = test_data
        self.global_mean = global_mean
        self.method = method
        self.partitions = k

    def calculate_predictions(self):
        if self.method == 'movie_genres' or self.method == 'movie_reviews':
            return self.item_based_knn()
        else: #user_data
            return self.user_based_knn()

    def user_based_knn(self):
        user_data_matrix = utils.read_user_data_file()
        similarity_matrix = 1 - pairwise_distances(user_data_matrix, metric='jaccard')
        predictions = []
        for data in self.test_data:
            test_id = data[0]
            user_id = data[1]
            movie_id = data[2]

            movie_ratings = np.array(self.train_rating_matrix[:,movie_id])
            user_ratings = self.train_rating_matrix[user_id]

            users_rating_movie = np.nonzero(movie_ratings)[0]
            user_similarities = similarity_matrix[user_id]
            user_similarities[user_id] = -1.0
            prediction = utils.knn_prediction(movie_ratings, users_rating_movie, user_similarities,user_ratings, self.partitions, self.global_mean)
            predictions.append([test_id, str(prediction)])
            print test_id, user_id, movie_id, prediction

        return predictions

    def generate_tfidf_feature_reviews(self):
        print 'Calculating Tf-Idf movie matrix'
        corpus = []
        movie_reviews = utils.read_movie_reviews()
        indices = dict()
        for index, movie in enumerate(movie_reviews):
            corpus.append(movie_reviews[movie])
            indices[movie] = index

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)
        tfidf_matrix = tfidf_matrix.toarray()

        nfeatures = tfidf_matrix.shape[1]

        nmovies = conf.nmovies

        movie_review_matrix = []
        for movie in range(nmovies):
            if movie in movie_reviews:
                index = indices[movie]
                movie_array = tfidf_matrix[index]
            else:
                movie_array = np.repeat(-9.9, nfeatures)
            movie_review_matrix.append(movie_array)

        return np.array(movie_review_matrix)


    def item_based_knn(self):
        if self.method == 'movie_genres':
            movie_matrix = utils.read_movie_data_file()
            similarity_matrix = 1 - pairwise_distances(movie_matrix, metric='jaccard')
            movies_with_reviews = None
        else:
            movie_matrix = self.generate_tfidf_feature_reviews()
            similarity_matrix = 1 - pairwise_distances(movie_matrix, metric='cosine')
            movies_with_reviews = utils.get_movies_with_reviews()

        predictions = []

        for data in self.test_data:
            test_id = data[0]
            user_id = data[1]
            movie_id = data[2]

            user_ratings = np.array(self.train_rating_matrix[user_id])
            movie_ratings = self.train_rating_matrix[:, movie_id]
            movies_rated_by_user = np.nonzero(user_ratings)[0]
            movie_similarities = similarity_matrix[movie_id]
            movie_similarities[movie_id] = -1.0

            if self.method == 'movie_genres':
                prediction = utils.knn_prediction(user_ratings, movies_rated_by_user, movie_similarities, movie_ratings, self.partitions, self.global_mean)
            else:# movie reviews
                movie_has_reviews = movie_id in movies_with_reviews
                prediction = utils.knn_prediction_v2(user_ratings, movies_rated_by_user, movie_similarities, movie_ratings, self.partitions, self.global_mean, movie_has_reviews)

            predictions.append([test_id, str(prediction)])
            print test_id, user_id, movie_id, prediction
        return predictions
