'''
Modulo principal do sistema
Neste modulo podem ser chamadas todos os algoritmo de recomendacao que foram implementados
'''

import recomender
import configuration as conf

if __name__ == '__main__':

    '''
     Path do arquivo de saida onde as predicoes serao armazenadas. 
    '''
    output_file = conf.path_submit_file




    '''
    Execucao do sistema de recomendacao
    Aqui pode ser escolhido o algoritmo que deseja-se avaliar com os seus parametros correspondentes  
    Aqui sao mostrados exemplos de como a chamada para cada algoritmo pode ser feita  
    '''

    system = recomender.SystemManager(method='items', k=4)

    #system = recomender.SystemManager(method='probabilistic')

    #system = recomender.SystemManager(method='baseline', l1=3, l2=2)

    #system = recomender.SystemManager(method='svd', k=1000)

    #system = recomender.SystemManager(method='gradient', sub_method='optimizedSVD', k=10, iter=20, lr=0.05, reg=0.02)
    #sub_method parameter could be 'funkSVD' or 'optimizedSVD'

    #system = recomender.SystemManager(method='knn', sub_method='movie_genres', k=2)
    #sub_method parameter could be 'movie_genres'  or 'movie_reviews' or 'user_data'

    #system = recomender.SystemManager(method='linearCBF', iter=40, lr=0.05, reg=0.02)

    #system = recomender.SystemManager(method='hybrid', sub_method='monolithic', k=3)
    #sub_method parameter could be 'monolithic' or 'parallel' or 'meta_parallel'

    #system = recomender.SystemManager(method='w2v', sub_method='parallel', vsize=100, k=5)
    #sub_method parameter could be 'parallel' or 'user_sim' or 'movie_sim'




    '''
    Depois da etapa de execucao do sistema, o arquivo .csv fica pronto para ser submetido no Kaggle
    '''
    system.execute(output_file)




