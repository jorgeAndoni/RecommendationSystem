import baseline
import utils
import numpy as np
import time
import svd_learning
import fbc_knn
import hybrid
import word2vec_method
import items
import probabilistic

class SystemManager(object):
    '''
    A classe faze a coneccao com todos os algoritmos de recomendacao que foram implementados
    '''

    def __init__(self, method='', sub_method='', k=50, l1=0, l2=0, iter=10, lr=0.05, reg=0.02, vsize=100): 
        self.start_time = time.time()
        self.method = method
        self.sub_method = sub_method
        self.k = k
        self.train_data = utils.read_train_file()
        self.test_data = utils.read_test_file()
        self.train_rating_matrix = utils.create_rating_matrix(self.train_data)
        self.global_mean = np.mean(self.train_data[:,2])
        self.lambda1 = l1
        self.lambda2 = l2
        self.niterations = iter
        self.lr = lr
        self.reg = reg
        self.w2v_vector_size = vsize

    def execute(self, output):
        if self.method == 'baseline':
            print 'Testing with baseline method'
            system_obj = baseline.BaselineMethod(self.train_rating_matrix, self.test_data, self.global_mean, self.lambda1, self.lambda2)
        elif self.method == 'items':
            print 'Testing with CF based on items'
            system_obj = items.CFItemsMethod(self.train_rating_matrix, self.test_data, self.global_mean, self.k)
        elif self.method == 'probabilistic':
            print 'Testing with probabilistic method'
            system_obj = probabilistic.ProbabilisticMethod(self.train_rating_matrix, self.test_data, self.global_mean)
        elif self.method == 'svd':
            print 'Testing with SVD method'
            system_obj = learning_methods.SVDMethod(self.train_rating_matrix, self.test_data, self.global_mean, self.k)
        elif self.method == 'gradient':
            print 'Testing with gradient method'
            system_obj = learning_methods.StochasticGradientMethod(self.train_data, self.train_rating_matrix, self.test_data, self.global_mean, self.sub_method, self.k, self.niterations, self.lr, self.reg)
        elif self.method == 'linearCBF':
            print 'Testing with LinearCBF method'
            features = utils.read_movie_data_file()
            system_obj = learning_methods.LinearCBFMethod(self.train_data, self.test_data, features, self.global_mean, self.niterations, self.lr, self.reg)
        elif self.method == 'knn':
            print 'Testing with FBC-knn methods'
            system_obj = fbc_knn.FBCKnnMethod(self.train_rating_matrix, self.test_data, self.global_mean, self.sub_method, self.k)
        elif self.method == 'hybrid':
            print 'Testing with hybrid methods'
            system_obj = hybrid.HybridMethod(self.train_rating_matrix, self.test_data, self.global_mean, self.sub_method, self.k)
        else:
            print 'Testing with Word2Vec method'
            system_obj = word2vec_method.Word2vecMethod(self.train_data, self.train_rating_matrix, self.test_data, self.global_mean, self.sub_method, self.w2v_vector_size, self.k)

        predictions = system_obj.calculate_predictions()
        self.finish_time = time.time()
        print 'Execution Time: ', self.finish_time-self.start_time
        utils.save_predictions(output, predictions)
    