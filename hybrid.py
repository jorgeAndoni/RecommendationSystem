import utils
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
import configuration as conf
import numpy as np
import learning_methods

class HybridMethod(object):

    '''
    Metodo hybrido e suas tres variacoes (Monolitico, parelela, meta-nivel + paralela)
    '''

    def __init__(self, train_rating_matrix, test_data, global_mean, sub_method, k):
        self.train_rating_matrix = train_rating_matrix
        self.test_data = test_data
        self.global_mean = global_mean
        self.method = sub_method
        self.partitions = k

    def calculate_predictions(self):
        if self.method == 'monolithic':
            return self.monolithic_hybridization()
        elif self.method == 'parallel':
            return self.parallel_hybridization()
        else: #meta_parallel
            return self.metaLevel_parallel_hybridization()


    def get_keyword_reviews_matrix(self):

        print 'Calculating Tf-Idf movie matrix'
        corpus = []
        movie_reviews = utils.read_movie_reviews()
        movie_indices = dict()
        for index, movie in enumerate(movie_reviews):
            corpus.append(movie_reviews[movie])
            movie_indices[movie] = index

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)
        tfidf_matrix = tfidf_matrix.toarray()

        vocabulary = vectorizer.get_feature_names() #55828 words

        keyword_dictionary = dict()
        all_keyword_lists = []
        index = 0
        for vector in tfidf_matrix:
            top_keyword_indices = (-vector).argsort()[:20]
            top_keyword_list = []
            for i in top_keyword_indices:
                keyword = vocabulary[i]
                top_keyword_list.append(keyword)
                if keyword in keyword_dictionary:
                    pass
                else:
                    keyword_dictionary[keyword] = index
                    index+=1
            all_keyword_lists.append(top_keyword_list)

        nfeatures = len(keyword_dictionary)

        nmovies = conf.nmovies
        movie_review_matrix = []
        for movie in range(nmovies):
            movie_array = np.zeros(nfeatures)
            if movie in movie_reviews:
                index = movie_indices[movie]
                movie_keywords = all_keyword_lists[index]
                for keyword in movie_keywords:
                    position = keyword_dictionary[keyword]
                    movie_array[position] = 1
            movie_review_matrix.append(movie_array)

        return np.array(movie_review_matrix)


    def monolithic_hybridization(self):
        genres_matrix = utils.read_movie_data_file()
        similarity_genres_matrix = 1 - pairwise_distances(genres_matrix, metric='jaccard')

        reviews_keyword_matrix = self.get_keyword_reviews_matrix()
        similarity_reviews_matrix = 1 - pairwise_distances(reviews_keyword_matrix, metric='jaccard')
        similarity_reviews_matrix = np.where(np.isnan(similarity_reviews_matrix), similarity_genres_matrix, similarity_reviews_matrix)

        similarity_movie_matrix = (similarity_genres_matrix+similarity_reviews_matrix)/2

        predictions = []
        for data in self.test_data:
            test_id = data[0]
            user_id = data[1]
            movie_id = data[2]

            user_ratings = np.array(self.train_rating_matrix[user_id])
            movie_ratings = self.train_rating_matrix[:, movie_id]

            movies_rated_by_user = np.nonzero(user_ratings)[0]
            movie_similarities = similarity_movie_matrix[movie_id]
            movie_similarities[movie_id] = -1.0
            prediction = utils.knn_prediction(user_ratings, movies_rated_by_user, movie_similarities, movie_ratings, self.partitions, self.global_mean)

            predictions.append([test_id, str(prediction)])
            print test_id, user_id, movie_id, prediction

        return predictions


    def parallel_hybridization(self):
        user_data_matrix = utils.read_user_data_file()
        similarity_user_matrix = 1 - pairwise_distances(user_data_matrix, metric='jaccard')

        genres_matrix = utils.read_movie_data_file()
        similarity_genres_matrix = 1 - pairwise_distances(genres_matrix, metric='jaccard')

        reviews_keyword_matrix = self.get_keyword_reviews_matrix()
        similarity_reviews_matrix = 1 - pairwise_distances(reviews_keyword_matrix, metric='jaccard')
        similarity_reviews_matrix = np.where(np.isnan(similarity_reviews_matrix), similarity_genres_matrix, similarity_reviews_matrix)

        predictions = []
        for data in self.test_data:
            test_id = data[0]
            user_id = data[1]
            movie_id = data[2]

            user_ratings = np.array(self.train_rating_matrix[user_id])
            movie_ratings = np.array(self.train_rating_matrix[:,movie_id])

            users_rating_movie = np.nonzero(movie_ratings)[0]
            movies_rated_by_user = np.nonzero(user_ratings)[0]

            user_similarities = similarity_user_matrix[user_id]
            user_similarities[user_id] = -1.0

            movie_genres_similarities = similarity_genres_matrix[movie_id]
            movie_genres_similarities[movie_id] = -1.0

            movie_reviews_similarities = similarity_reviews_matrix[movie_id]
            movie_reviews_similarities[movie_id] = -1.0

            user_data_prediction = utils.knn_prediction(movie_ratings, users_rating_movie, user_similarities, user_ratings, self.partitions,self.global_mean)
            movie_genres_prediction = utils.knn_prediction(user_ratings, movies_rated_by_user, movie_genres_similarities, movie_ratings, self.partitions,self.global_mean)
            movie_reviews_prediction = utils.knn_prediction(user_ratings, movies_rated_by_user, movie_reviews_similarities, movie_ratings, self.partitions,self.global_mean)

            #prediction = (user_data_prediction*0.65) + (movie_genres_prediction*0.2) + (movie_reviews_prediction*0.15)
            prediction = (user_data_prediction * 0.34) + (movie_genres_prediction * 0.33) + (movie_reviews_prediction * 0.33)
            predictions.append([test_id, str(prediction)])
            print test_id, user_id, movie_id, prediction

        return predictions


    def metaLevel_parallel_hybridization(self):

        svd_matrix = utils.load_data_from_disk(conf.path_SVD_P_Q_matrix) # aqui carrego a matriz gradiente que obteve os melhores resultados
        user_matrix = svd_matrix[0]
        movie_matrix = svd_matrix[1]

        # O codigo comentado usa o metodo SVD da algebra linear para calcular as matrizes P e Q
        #obj_svd = learning_methods.SVDMethod(self.train_rating_matrix, self.test_data, self.global_mean,2000)
        #svd_matrix = obj_svd.generate_svd_matrix()
        #user_matrix = svd_matrix[0]
        #movie_matrix = svd_matrix[1]

        similarity_users = 1 - pairwise_distances(user_matrix, metric="cosine")
        similarity_movies = 1 - pairwise_distances(movie_matrix, metric="cosine")

        predictions = []
        for data in self.test_data:
            test_id = data[0]
            user_id = data[1]
            movie_id = data[2]

            user_vector = np.array(self.train_rating_matrix[user_id])
            movies_rated_by_user = np.nonzero(user_vector)[0]
            movie_vector = np.array(self.train_rating_matrix[:, movie_id])
            users_rating_movie = np.nonzero(movie_vector)[0]

            user_similarities = similarity_users[user_id]
            movie_similarities = similarity_movies[movie_id]
            user_similarities[user_id] = -1.0
            movie_similarities[movie_id] = -1.0


            prediction_1 = utils.knn_prediction(user_vector, movies_rated_by_user, movie_similarities, movie_vector, self.partitions, self.global_mean)
            prediction_2 = utils.knn_prediction(movie_vector, users_rating_movie, user_similarities, user_vector, self.partitions, self.global_mean)


            prediction = (0.5 * prediction_1) + (0.5 * prediction_2)
            predictions.append([test_id, str(prediction)])
            print test_id, user_id, movie_id, prediction

        return predictions
