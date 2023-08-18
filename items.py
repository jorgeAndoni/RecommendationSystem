from sklearn.metrics import pairwise_distances
import numpy as np
import utils

class CFItemsMethod(object):
    '''
    Metodo baseado na vizinhanza de itens
    '''

    def __init__(self, train_rating_matrix, test_data, global_mean, k):
        self.train_rating_matrix = train_rating_matrix
        self.test_data = test_data
        self.global_mean = global_mean
        self.partitions = k

    def calculate_predictions(self):
        matrix  = self.train_rating_matrix.transpose()
        #sim_matrix = 1 - pairwise_distances(matrix, metric='cosine')
        sim_matrix = 1 - pairwise_distances(matrix, metric='correlation')
        sim_matrix = np.where(np.isnan(sim_matrix), -1.0, sim_matrix)

        predictions = []
        for data in self.test_data:
            test_id = data[0]
            user_id = data[1]
            movie_id = data[2]

            user_ratings = np.array(self.train_rating_matrix[user_id])
            movie_ratings = np.array(self.train_rating_matrix[:, movie_id])
            movies_rated_by_user = np.nonzero(user_ratings)[0]

            movie_similarities = sim_matrix[movie_id]
            movie_similarities[movie_id] = -1.0

            prediction = utils.knn_prediction(user_ratings, movies_rated_by_user, movie_similarities, movie_ratings, self.partitions, self.global_mean)

            print test_id, user_id, movie_id, prediction
            predictions.append([test_id, str(prediction)])

        return predictions
