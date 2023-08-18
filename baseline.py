import numpy as np
import configuration as conf

class BaselineMethod(object):

    '''
    Metodo baseline 
    '''

    def __init__(self, train_rating_matrix, test_data, global_mean, l1=0, l2=0):
        self.train_rating_matrix = train_rating_matrix
        self.test_data = test_data
        self.global_mean = global_mean
        self.lambda1 = l1
        self.lambda2 = l2

    def calculate_user_item_bias(self):
        nmovies = conf.nmovies
        nusers = conf.nusers
        bi_movies = np.zeros(nmovies)
        bu_users = np.zeros(nusers)

        for i in range(nmovies):
            movie_ratings = self.train_rating_matrix[:, i]
            nonzero = np.count_nonzero(movie_ratings)
            indices_valid_ratings = np.nonzero(movie_ratings)[0]
            summation = 0.0
            if len(indices_valid_ratings) != 0:
                for index in indices_valid_ratings:
                    summation += movie_ratings[index] - self.global_mean
                bi = summation / float(nonzero + self.lambda1)
            else:
                bi = float('NaN')
            bi_movies[i] = bi

        for i in range(nusers):
            user_ratings = self.train_rating_matrix[i]
            nonzero = np.count_nonzero(user_ratings)
            indices_valid_ratings = np.nonzero(user_ratings)[0]
            summation = 0.0
            if len(indices_valid_ratings) != 0:
                for index in indices_valid_ratings:
                    summation += user_ratings[index] - self.global_mean - bi_movies[index]
                bu = summation / float(nonzero + self.lambda2)
            else:
                bu = float('NaN')
            bu_users[i] = bu

        return [bu_users, bi_movies]


    def calculate_predictions(self):
        bias_data = self.calculate_user_item_bias()
        bu_users = bias_data[0]
        bi_movies = bias_data[1]

        predictions = []
        for data in self.test_data:
            test_id = data[0]
            user_id = data[1]
            movie_id = data[2]
            prediction = self.global_mean + bu_users[user_id] + bi_movies[movie_id]
            if np.isnan(prediction):
                movie_ratings = self.train_rating_matrix[:,movie_id]
                denominator = np.count_nonzero(movie_ratings)
                if denominator!=0:
                    mean = np.sum(movie_ratings)/float(denominator)
                else:
                    mean = self.global_mean
                prediction = mean
            print test_id, user_id, movie_id, prediction
            predictions.append([test_id, str(prediction)])
        return predictions



