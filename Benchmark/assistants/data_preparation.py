from pandas import read_csv
from sklearn.utils import shuffle

class DataPreparation:
    def read_data(trainfile, testfile, augfile=None):
        train = read_csv(trainfile)
        train.columns = ['text', 'label']
        test = read_csv(testfile)
        test.columns = ['text', 'label']

        if augfile is not None:
            aug = read_csv(augfile)
            aug.columns = ['text', 'label']
            train = train.append(aug, ignore_index=True)
            print('shuffle')
            train = shuffle(train)
        
        test = test.drop_duplicates().dropna()
        train = train.drop_duplicates().dropna()
        
        print('Train size:', train.shape[0])
        print('Test size:', test.shape[0])

        return train.text, test.text, train.label, test.label