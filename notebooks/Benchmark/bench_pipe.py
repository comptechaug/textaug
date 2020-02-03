from assistants.data_preparation import DataPreparation
from assistants.embeddings import Embeddings
from assistants.classifiers import Classifiers, Metrics
import os.path
import os
import sys


def benchmark_pipeline(datafolder, trainfile, testfile, augfile):

    datafile = datafolder + trainfile
    print('Reading...', datafile)
    aug = datafolder+augfile if augfile is not None else None
    data = DataPreparation.read_data(datafolder + trainfile, datafolder+testfile,  aug)
    
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
        "Naive Bayes" : Classifiers.naive_bayes,
        "Neural network" : Classifiers.perceptron,
        "XGboost" : Classifiers.xgboost,
    }
    results = dict()

    print('Classification time...')
    for kind in data_encoded:
        for clf in dict_classifiers:
            try:
                accuracy, balanced_accuracy, f1, conf_mat = dict_classifiers[clf](*data_encoded[kind])  
                print('+++', datafile, '|', clf, '|', kind, '|')
                print(f'\tAccuracy {accuracy:.3f}; Balanced accuracy {balanced_accuracy:.3f}; F1-measure weighted {f1:.3f}.')
                print('Confusion matrix\n', conf_mat)
            except:
                print('ERROR.', datafile, '|', clf, '|', kind, '|')

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
    
    folder = sys.argv[1]
    aug = sys.argv[2] if len(sys.argv)>2 else ''
    print(aug)
    print('Data folder', folder)

    testfile = '/test.csv'
    trainfile = '/train.csv'
    augfile = ('/'+aug+'.csv') if os.path.exists(folder+'/' + aug + '.csv') else None
    print('with augmentation') if (augfile is not None) else print('without augmentation')

    benchmark_pipeline(folder, trainfile, testfile, augfile )