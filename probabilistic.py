import configuration as conf
import numpy as np
from collections import Counter

class ProbabilisticMethod(object):

    '''
    Metodo probabilistico
    '''

    def __init__(self, train_rating_matrix, test_data, global_mean):
        self.train_rating_matrix = train_rating_matrix
        self.test_data = test_data
        self.global_mean = global_mean

    def calculate_predictions(self):
        nmovies = conf.nmovies

        movie_counts_probs = dict()
        for i in range(nmovies):
            movie_ratings = self.train_rating_matrix[:,i]
            all_counts = Counter(movie_ratings)
            nonzero = np.count_nonzero(movie_ratings)
            probabilities = []
            for j in range(5):
                try:
                    prob = all_counts[j+1]/float(nonzero)
                    probabilities.append(prob)
                except ZeroDivisionError:
                    pass
            movie_counts_probs[i] = [all_counts, probabilities]


        predictions = []
        for data in self.test_data:
            test_id = data[0]
            user_id = data[1]
            movie_id = data[2]
            user_ratings = np.array(self.train_rating_matrix[user_id])
            movie_ratings = np.array(self.train_rating_matrix[:, movie_id])

            user_has_validations = np.any(user_ratings)
            movie_has_validations = np.any(movie_ratings)


            if movie_has_validations == False or user_has_validations == False:
                prediction = self.global_mean
            else:
                movie_count_prob_data = movie_counts_probs[movie_id]
                rating_counts = movie_count_prob_data[0]
                py_probabilities = movie_count_prob_data[1]

                conditionals = []
                for rating in range(1,6):
                    product = 1
                    for i in range(nmovies):
                        item = self.train_rating_matrix[:,i]
                        numerator = len(np.intersect1d(np.where(movie_ratings==rating), np.where(item==user_ratings[i])))+0.0001
                        denominator = rating_counts[rating]
                        value = 0
                        if denominator!=0:
                            value = numerator/denominator
                        product*=value
                    cond_prob = py_probabilities[rating - 1] * product
                    conditionals.append(cond_prob)

                if np.any(conditionals):
                    prediction = np.argmax(conditionals)+1
                else:
                    prediction = np.sum(movie_ratings)/float(np.count_nonzero(movie_ratings))
            predictions.append([test_id, str(prediction)])
            print test_id, user_id, movie_id, prediction

        return predictions
