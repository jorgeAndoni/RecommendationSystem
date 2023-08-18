from gensim.models import Word2Vec
import configuration as conf
import numpy as np
from sklearn.metrics import pairwise_distances
import utils

class Word2vecMethod(object):

    '''
    Algortimo baseado no Word2Vec  que gera matrizes de similaridade de usuarios e items. Tambem usa uma abordagem hybrida paralela
    '''

    def __init__(self, train_data, train_rating_matrix, test_data, global_mean, method, w2v_size, k):
        self.train_data = train_data
        self.train_rating_matrix = train_rating_matrix
        self.test_data = test_data
        self.global_mean = global_mean
        self.method = method
        self.v_size = w2v_size
        self.partitions = k


    def training(self):
        print 'Training word2vec based method'
        user_ratings = dict()

        for data in self.train_data:
            user_id = data[0]
            movie_id = data[1]
            rating = data[2]
            if rating>=3:
                if user_id in user_ratings:
                    user_ratings[user_id].append(str(movie_id))
                else:
                    user_ratings[user_id] = [str(movie_id)]

        corpus = []
        for i in user_ratings:
            corpus.append(user_ratings[i])


        model = Word2Vec(corpus, size=self.v_size, window=5, min_count=1, workers=4)

        nusers = conf.nusers
        nmovies = conf.nmovies

        movie_arrays = []
        for i in range(nmovies):
            movie_id = str(i)
            if movie_id in model:
                vector = model.wv[movie_id]
            else:
                vector = np.repeat(-9.9, self.v_size)
            movie_arrays.append(vector)


        user_arrays = []
        for i in range(nusers):
            user_id = i
            if user_id in user_ratings:
                items = user_ratings[user_id]
                vector = utils.get_w2v_vector(model, items)
            else:
                vector = np.repeat(-9.9, self.v_size)
            user_arrays.append(vector)

        return [user_arrays, movie_arrays]


    def calculate_predictions(self):
        data = self.training()
        user_arrays = data[0]
        movie_arrays = data[1]

        similarity_matrix_users = 1 - pairwise_distances(user_arrays, metric="cosine")
        similarity_matrix_movies = 1 - pairwise_distances(movie_arrays, metric="cosine")

        predictions = []
        for data in self.test_data:
            test_id = data[0]
            user_id = data[1]
            movie_id = data[2]

            user_ratings = np.array(self.train_rating_matrix[user_id])
            movie_ratings = np.array(self.train_rating_matrix[:,movie_id])

            users_rating_movie = np.nonzero(movie_ratings)[0]
            movies_rated_by_user = np.nonzero(user_ratings)[0]

            user_similarities = similarity_matrix_users[user_id]
            user_similarities[user_id] = -1.0

            movie_similarities = similarity_matrix_movies[movie_id]
            movie_similarities[movie_id] = -1.0

            if self.method == 'user_sim':
                prediction = utils.knn_prediction(movie_ratings, users_rating_movie, user_similarities, user_ratings,self.partitions,self.global_mean)
            elif self.method == 'movie_sim':
                prediction = utils.knn_prediction(user_ratings, movies_rated_by_user, movie_similarities,movie_ratings,self.partitions,self.global_mean)
            else:#parallel
                user_prediction = utils.knn_prediction(movie_ratings, users_rating_movie, user_similarities, user_ratings,self.partitions,self.global_mean)
                movie_prediction = utils.knn_prediction(user_ratings, movies_rated_by_user, movie_similarities, movie_ratings,self.partitions,self.global_mean)
                prediction = (0.5 * user_prediction) + (0.5 * movie_prediction)

            predictions.append([test_id, str(prediction)])
            print test_id, user_id, movie_id, prediction

        return predictions
