from assistants.data_preparation import DataPreparation
from assistants.embeddings import Embeddings
from assistants.classifiers import Classifiers, Metrics
import os.path


def benchmark_pipeline(datafolder, trainfile, testfile):

    datafile = datafolder + trainfile
    print('Reading...', datafile)
    data = DataPreparation.read_data(datafolder + trainfile, datafolder+testfile)
    
    print('Preparation to embedding')
    embed = Embeddings()
    embed.set_data(*data)
    
    dict_embeddings = {
        "TF-IDF" : embed.tf_idf_vectorizer,
        "Fasttext" : embed.fasttext_vectorizer,
    }
    data_encoded = dict()
    
    for kind in dict_embeddings:
        print(kind, 'embedding...')
        data_encoded[kind] = dict_embeddings[kind]()
    embed.fasttext_model_clear_from_ram()

    dict_classifiers = {
        "Log regression" : Classifiers.log_reg,
        # "Random forest" : Classifiers.random_forest,
        # "SVC" : Classifiers.svc,
        # "PCA & Log reg" : Classifiers.pca_log_reg,
        "Naive Bayes" : Classifiers.naive_bayes,
        "K Neighbors" : Classifiers.k_neighbors,
        "Neural network" : Classifiers.perceptron,
    }
    results = dict()

    print('Classification time...')
    for kind in data_encoded:
        for clf in dict_classifiers:
            try:
                score = dict_classifiers[clf](*data_encoded[kind])
                print('+++', datafile, 'Accuracy score |', clf, '|', kind, '|', score)
            except:
                print('ERROR.', datafile, 'Accuracy score |', clf, '|', kind)

    print('Bye')
    return

if __name__=="__main__":
    if os.path.isfile('cc.ru.300.bin'):
        print('Fasttext model was found.')
    else:
        print('Please, download fasttext model using these commands')
        print('wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.bin.gz')
        print('gunzip cc.ru.300.bin.gz')
        raise DictionaryError
    
    data_list = [
        '../../data/twitts/',
        '../../data/SentEvalRu/Poems classifier/', 
        '../../data/SentEvalRu/Proza classifier/', 
        '../../data/SentEvalRu/Readability classifier/',
        '../../data/SentEvalRu/Tags classifier/'
        ]

    testfile = 'test.csv'
    trainfile = 'train.csv'
             
    for datafile in data_list:
        benchmark_pipeline(datafie, trainfile, testfile )