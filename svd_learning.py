import numpy as np
import configuration as conf
import utils
from random import shuffle
import baseline


class SVDMethod(object):

    '''
    Metodo baseado no SVD da Algebra Linear
    '''

    def __init__(self, rating_matrix, test_data, global_mean, k=50):
        self.train_rating_matrix = rating_matrix
        self.test_data = test_data
        self.global_mean = global_mean
        self.k = k

    def generate_svd_matrix(self):
        print 'Calculating SVD matrix'
        P, S, Q = np.linalg.svd(self.train_rating_matrix, full_matrices=True)
        Q = Q.transpose()
        P2 = P[:, 0:self.k]
        Q2 = Q[:, 0:self.k]
        S2 = S[0:self.k]
        S2 = np.diag(S2)
        return [P2, Q2, S2]


    def calculate_predictions(self):
        matrix_data = self.generate_svd_matrix()
        P2 = matrix_data[0]
        Q2 = matrix_data[1]
        S2 = matrix_data[2]

        obj_bies = baseline.BaselineMethod(self.train_rating_matrix, self.test_data, self.global_mean)
        bies_data = obj_bies.calculate_user_item_bias()

        bu_users = bies_data[0]
        bi_movies = bies_data[1]

        predictions = []
        for data in self.test_data:
            test_id = data[0]
            user_id = data[1]
            movie_id = data[2]
            user_has_validations = np.any(self.train_rating_matrix[user_id])
            movie_has_validations = np.any(self.train_rating_matrix[:, movie_id])
            if user_has_validations and movie_has_validations:
                bui = self.global_mean + bu_users[user_id] + bi_movies[movie_id]
                auxiliar = np.dot(P2[user_id], S2)
                prediction = bui + np.dot(auxiliar, Q2[movie_id])

            else:
                if movie_has_validations:
                    movie_ratings = self.train_rating_matrix[:, movie_id]
                    denominator = np.count_nonzero(movie_ratings)
                    prediction = np.sum(movie_ratings)/float(denominator)
                else:
                    prediction = self.global_mean
            predictions.append([test_id, str(prediction)])
            print test_id, user_id, movie_id, prediction

        return predictions


class StochasticGradientMethod(object):

    '''
    Metodo baseado no gradiente descendente estocastico
    '''

    def __init__(self, train_data, rating_matrix, test_data, global_mean, sub_method, k=20, iter=10, lr=0.05, reg=0.02):
        self.train_data = train_data
        self.train_rating_matrix = rating_matrix
        self.test_data = test_data
        self.global_mean = global_mean
        self.method = sub_method
        self.k = k
        self.iter = iter
        self.lr = lr
        self.reg = reg


    def funkSVD(self):
        print 'Training funk SVD ...'
        nusers = conf.nusers
        nitems = conf.nmovies

        bu = np.zeros(nusers)
        bi = np.zeros(nitems)

        P = np.full((nusers, self.k), 0.1)
        Q = np.full((nitems, self.k), 0.1)

        sample_k = [x for x in range(self.k)]
        shuffle(sample_k)

        for iteration, f in enumerate(sample_k):
            print 'Iterating k', iteration + 1
            for l in range(self.iter):
                for j in range(len(self.train_data)):
                    u = self.train_data[j, 0]
                    i = self.train_data[j, 1]
                    r_ui = self.train_data[j, 2]
                    pred = self.global_mean + bu[u] + bi[i] + (np.matmul(P[u], Q[i]))
                    e_ui = r_ui - pred
                    bu[u] = bu[u] + self.lr * e_ui
                    bi[i] = bi[i] + self.lr * e_ui
                    temp_uf = P[u, f]
                    P[u, f] = P[u, f] + self.lr * (e_ui * Q[i, f] - self.reg * P[u, f])
                    Q[i, f] = Q[i, f] + self.lr * (e_ui * temp_uf - self.reg * Q[i, f])
        return [bu, bi, P, Q]


    def optimizedSVD(self):
        print 'Training optimized SVD ...'
        nusers = conf.nusers
        nitems = conf.nmovies

        bu = np.zeros(nusers)
        bi = np.zeros(nitems)

        P = np.random.normal(0, 0.1, (nusers, self.k))
        Q = np.random.normal(0, 0.1, (nitems, self.k))

        for l in range(self.iter):
            print 'Iteration', l+1
            for j in range(len(self.train_data)):
                u = self.train_data[j,0]
                i = self.train_data[j,1]
                r_ui = self.train_data[j,2]
                pred = self.global_mean + bu[u] + bi[i] + np.dot(P[u], Q[i])
                e_ui = r_ui-pred
                bu[u] = bu[u] + self.lr * e_ui
                bi[i] = bi[i] + self.lr * e_ui
                for f in range(self.k):
                    temp_uf = P[u,f]
                    P[u,f] = P[u,f] + self.lr * (e_ui*Q[i,f]-self.reg*P[u,f])
                    Q[i,f] = Q[i,f] + self.lr * (e_ui*temp_uf-self.reg*Q[i,f])
        return [bu, bi, P, Q]


    def calculate_predictions(self):
        if self.method == 'funkSVD':
            matrix_data = self.funkSVD()
        else:
            matrix_data = self.optimizedSVD()

        bu = matrix_data[0]
        bi = matrix_data[1]
        P = matrix_data[2]
        Q = matrix_data[3]
        utils.write_data_to_disk(conf.path_SVD_P_Q_matrix, [P,Q])

        predictions = []
        for data in self.test_data:
            test_id = data[0]
            user_id = data[1]
            movie_id = data[2]
            user_has_validations = np.any(self.train_rating_matrix[user_id])
            movie_has_validations = np.any(self.train_rating_matrix[:,movie_id])
            if user_has_validations and movie_has_validations:
                prediction = self.global_mean + bu[user_id] + bi[movie_id] + np.dot(P[user_id], Q[movie_id])
            else:
                if movie_has_validations:
                    movie_ratings = self.train_rating_matrix[:, movie_id]
                    denominator = np.count_nonzero(movie_ratings)
                    prediction = np.sum(movie_ratings)/float(denominator)
                else:
                    prediction = self.global_mean

            predictions.append([test_id, str(prediction)])
            print test_id, user_id, movie_id, prediction
        return predictions



class LinearCBFMethod(object):

    '''
    Metodo baseado no LinearCBF que usa os generos dos filmes como features
    '''

    def __init__(self, train_data, test_data, features, global_mean, iter=10, lr=0.05, reg=0.02):
        self.train_data = train_data
        self.test_data = test_data
        self.features = features
        self.global_mean = global_mean
        self.iter = iter
        self.lr = lr
        self.reg = reg

    def training(self):
        print 'Training linear model ....'
        nusers = conf.nusers
        nitems = conf.nmovies
        nfeatures = self.features.shape[1]
        insert = np.ones(nitems)
        self.features = np.column_stack((self.features, insert))
        profiles = np.random.normal(0, 0.1, (nusers, nfeatures + 1))
        for l in range(self.iter):
            print 'Iteration', l + 1
            for j in range(len(self.train_data)):
                u = self.train_data[j, 0]
                i = self.train_data[j, 1]
                r_ui = self.train_data[j, 2]
                e_ui = np.dot(profiles[u], self.features[i]) - r_ui
                for k in range(nfeatures):
                    profiles[u,k] = profiles[u,k] - self.lr * (e_ui*self.features[i,k]+self.reg*profiles[u,k])
                k = -1
                profiles[u,k] = profiles[u, k] - self.lr * (e_ui * self.features[i, k])
        return profiles

    def calculate_predictions(self):
        profiles = self.training()
        predictions = []
        for data in self.test_data:
            test_id = data[0]
            user_id = data[1]
            movie_id = data[2]
            prediction = np.dot(profiles[user_id], self.features[movie_id])
            if prediction<1:
                prediction = self.global_mean
            predictions.append([test_id, str(prediction)])
            print test_id, user_id, movie_id, prediction
        return predictions
