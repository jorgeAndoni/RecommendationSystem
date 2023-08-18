import configuration as conf
import pandas as pd
import numpy as np
import csv
import cPickle
from collections import Counter
import string
from nltk import word_tokenize
from  nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.corpus import wordnet

'''
Este modulo contem um conjunto de funcoes auxiliares que sao usadas pelos diferentes algoritmos de recomendacao
'''


# Funcoes de leitura e escrita de arquivos

def write_data_to_disk(file_path, data):
    '''
    Tem operacoes que levam muito tempo ser calculadas. O melhor seria guardar os resultados em memoria
    A funcao permite armazenar qualquer tipo de dato para depois ser usado 
    '''
    with open(file_path, 'wb') as fid:
        cPickle.dump(data, fid)


def load_data_from_disk(file):
    '''
    Funcao para carregar dados que foram armazenados na memoria
    '''
    print 'loading data from disk ....'
    with open(file, 'rb') as fid:
        data = cPickle.load(fid)
    print 'data loaded!'
    return data



# Algoritmos para a leitura dos dados de treinamento, testes, usuario e filmes
def read_train_file():
    '''
    Leitura do arquivo de treinamento
    '''
    path = conf.path_train_file
    data = pd.read_csv(path)
    user_ids = data['user_id']-1
    movie_ids = data['movie_id']-1
    ratings = data['rating']
    result = np.array(user_ids)
    result = np.column_stack((result, movie_ids))
    result = np.column_stack((result, ratings))
    return result

def read_test_file():
    '''
    Leitura deo arquivo de teste
    '''
    path = conf.path_test_file
    data = pd.read_csv(path)
    test_id = data['id']
    user_ids = data['user_id'] - 1
    movie_ids = data['movie_id'] - 1
    result = np.array(test_id)
    result = np.column_stack((result, user_ids))
    result = np.column_stack((result, movie_ids))
    return result


def create_rating_matrix(train_data):
    '''
    Criacao da matriz de  ratings (usuarios x filmes)
    '''
    matrix = np.zeros((conf.nusers, conf.nmovies))
    for data in train_data:
        user_id = data[0]
        movie_id = data[1]
        rating = data[2]
        matrix[user_id][movie_id] = rating
    return matrix

def read_movie_data_file():
    '''
    Leitura do arquivo de generos de filmes
     criacao dos vetores de filmes baseado nos generos
    '''
    path = conf.path_movies_file
    data = pd.read_csv(path)
    movie_ids = data['movie_id']-1
    movie_genres = data['genres']

    genres = dict()
    genres['Action'] = 0
    genres['Adventure'] = 1
    genres['Animation'] = 2
    genres['Children'] = 3
    genres['Comedy'] = 4
    genres['Crime'] = 5
    genres['Documentary'] = 6
    genres['Drama'] = 7
    genres['Fantasy'] = 8
    genres['Film-Noir'] = 9
    genres['Horror'] = 10
    genres['Musical'] = 11
    genres['Mystery'] = 12
    genres['Romance'] = 13
    genres['Sci-Fi'] = 14
    genres['Thriller'] = 15
    genres['War'] = 16
    genres['Western'] = 17

    movie_features = []

    for movie_id, gen in zip(movie_ids, movie_genres):
        mgenres = gen.split('|')
        movie_values = np.zeros(len(genres))
        for g in mgenres:
            index = genres[g]
            movie_values[index] = 1
        movie_features.append(movie_values)
    return np.array(movie_features)

def read_user_data_file():
    '''
    Leitura do arquivo de dados de usuarios
    Criacao dos vetores baseados nos dados dos usuarios
    '''
    path = conf.path_users_file
    data = pd.read_csv(path)
    user_ids = data['user_id']-1
    gender = data['gender']
    age = data['age']
    occupation = data['occupation']

    gender_dict = dict()
    gender_dict['F'] = 0
    gender_dict['M'] = 1

    age_dict = dict()
    age_dict[1] = 1
    age_dict[18] = 2
    age_dict[25] = 2
    age_dict[35] = 2
    age_dict[45] = 2
    age_dict[50] = 3
    age_dict[56] = 3

    user_features = []
    for us, gen, ag, occ in zip(user_ids, gender, age, occupation):
        user_array = np.zeros(25)
        user_array[0] = gender_dict[gen]
        age_position = age_dict[ag]
        user_array[age_position] = 1
        occupation_position = occ + 4
        user_array[occupation_position] = 1
        user_features.append(user_array)
    return np.array(user_features)



# Algoritmos de pre-procesamiento de texto para os reviews dos  filmes

def containsNonAscii(s):
    return any(ord(i)>127 for i in s)

def get_pos(word):
    w_synsets = wordnet.synsets(word)
    pos_counts = Counter()
    pos_counts["n"] = len([item for item in w_synsets if item.pos() == "n"])
    pos_counts["v"] = len([item for item in w_synsets if item.pos() == "v"])
    pos_counts["a"] = len([item for item in w_synsets if item.pos() == "a"])
    pos_counts["r"] = len([item for item in w_synsets if item.pos() == "r"])

    most_common_pos_list = pos_counts.most_common(3)
    return most_common_pos_list[0][0]


def text_preprocessing(text):
    '''
    Pre-procesamento de um texto que inclue normalizacao, eliminacao de stopworods e lematizacao
    '''
    content = text.lower()
    for c in string.punctuation:
        content = content.replace(c, " ")

    content = ''.join([i for i in content if not i.isdigit()])
    content = " ".join(content.split())

    content_list = word_tokenize(content)
    cleaned_words = [word for word in content_list if not containsNonAscii(word)]

    stopwords = dict()
    for i in ENGLISH_STOP_WORDS:
        stopwords[i] = 0

    text_without_sw = []
    for word in cleaned_words:
        if not word in stopwords and len(word) > 2:
            text_without_sw.append(word)

    text_lemmatized = []
    for word in text_without_sw:
        text_lemmatized.append(WordNetLemmatizer().lemmatize(word, get_pos(word)))

    return text_lemmatized



# Funcoes para o pre-processamento das revisoes dos filmes

def save_movie_reviews():
    '''
    Leitura do arquivo de revisoes dos filmes
    Pre-processamento das revisoes
    As revisoes sao armazenadas em memoria
    '''
    start_time = time.time()
    path = conf.path_movies_reviews_file
    data = pd.read_csv(path)
    movie_ids = data['movie_id']-1
    content = data['text']
    reviews = dict()
    for movie, review in zip(movie_ids, content):
        review = text_preprocessing(review)
        if movie in reviews:
            reviews[movie].append(review)
        else:
            reviews[movie] = [review]
    write_data_to_disk(output, reviews)



def get_movies_with_reviews(): 
    '''
    Permite verificar  quais sao os filmes que tem revisoes
    '''
    path = conf.path_movies_reviews_file
    data = pd.read_csv(path)
    movie_ids = data['movie_id'] - 1
    results = dict()
    for movie in movie_ids:
        if movie in results:
            pass
        else:
            results[movie] = 0
    return results


def read_movie_reviews():
    '''
    Permite carregar as revisoes pre-processadas que foram armazenadas em memoria
    '''
    path = conf.path_processed_movie_reviews
    movie_reviews = load_data_from_disk(path)
    movie_reviews_dict = dict()
    for movie in movie_reviews:
        reviews = movie_reviews[movie]
        str_reviews = ''
        for review in reviews:
            str_reviews+=  ' '.join(review) + ' '
        movie_reviews_dict[movie] =  str_reviews
    return movie_reviews_dict




# Funcoes auxiliares

def save_predictions(path, data):
    '''
    Permite armazenar as predicoes calculadas por algum algorimo em um arquivo do tipo .csv
    O arquivo fica pronto para ser avaliado no Kaggle
    '''
    titles = ['id', 'rating']
    myfile = open(path, 'wb')
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(titles)
    for prediction in data:
        wr.writerow(prediction)


def get_sorted_indices(array_list):
    '''
    Para obter os indices ordenados de um vetor
    '''
    return (-array_list).argsort()


def get_w2v_vector(model, items):
    '''
    Permite obter os vetores baseados no modelo de wor2vec
    '''
    w2v_vectors = []
    for item in items:
        vector = model.wv[item]
        w2v_vectors.append(vector)
    return np.mean(w2v_vectors, axis=0)


def knn_prediction(complete_rating_vector, valid_rating_indices, all_similarities, other_ratings, partitions, global_mean):
    '''
    Permite fazer a predicao baseada nos vizinhos mais proximos de um usuario ou filme
    '''
    indices_sorted_similarities = get_sorted_indices(all_similarities)
    k = len(valid_rating_indices) / partitions # ajuste do parametro k 
    numerator = 0.0
    denominator = 0.0
    if len(valid_rating_indices) != 0:
        sorted_valid_indices = []
        for i in indices_sorted_similarities:
            if i in valid_rating_indices:
                sorted_valid_indices.append(i)
        k_sorted = sorted_valid_indices[0:k]
        for index in k_sorted:
            numerator += all_similarities[index] * complete_rating_vector[index]
            denominator += all_similarities[index]
        if denominator != 0:
            prediction = numerator / denominator
        else:
            prediction = global_mean
    else:
        auxiliar = np.count_nonzero(other_ratings)
        if auxiliar != 0:
            prediction = np.sum(other_ratings) / float(auxiliar)
        else:
            prediction = global_mean
    return prediction


def knn_prediction_v2(complete_rating_vector, valid_rating_indices, all_similarities, other_ratings, partitions, global_mean, movie_has_reviews):
    '''
    Pequena modificacao para o algoritmo de predicao knn baseado nas revisoes dos filmes
    '''
    if movie_has_reviews:
        return knn_prediction(complete_rating_vector, valid_rating_indices, all_similarities, other_ratings, partitions, global_mean)
    auxiliar = np.count_nonzero(other_ratings)
    if auxiliar != 0:
        return np.sum(other_ratings) / float(auxiliar)
    return global_mean



if __name__ == '__main__':

    pass