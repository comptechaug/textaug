from pandas import read_csv

class DataPreparation:
    def read_data(trainfile, testfile):
        train = read_csv(trainfile)
        train.columns = ['text', 'label']
        test = read_csv(testfile)
        test.columns = ['text', 'label']
        
        test = test.drop_duplicates().dropna()
        train = train.drop_duplicates().dropna()
        
        print('Train size:', train.shape[0])
        print('Test size:', test.shape[0])

        return train.text, test.text, train.label, test.label