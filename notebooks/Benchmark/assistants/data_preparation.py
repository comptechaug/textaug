from sklearn.model_selection import train_test_split
from pandas import read_csv, DataFrame
import re

class DataPreparation:
    def tweets(data_folder):
        pos = read_csv(data_folder + '/twitts/positive.csv', sep=';', header=None)[[3]].assign(positive=1).rename({3:'tweet'}, axis=1)
        neg = read_csv(data_folder + '/twitts/negative.csv', sep=';', header=None)[[3]].assign(positive=0).rename({3:'tweet'}, axis=1)
        tweets = pos.append(neg, ignore_index=True)

        table_len = tweets.shape[0]
        tweets = tweets.drop_duplicates()
        print('Dublicates removed:', table_len - tweets.shape[0], 'from', table_len)

        return train_test_split(tweets.tweet, tweets.positive, test_size=0.33, random_state=42)

    def readability(data_folder, kind='Readability classifier'):
        train = read_csv(data_folder + '/SentEvalRu/'+ kind + '/train.csv', sep='\n', header=None)
        train = DataFrame( train[0].apply(lambda x: re.findall(r'(.*)\t(\d+$)', x)[0] ).tolist() )

        test = read_csv(data_folder + '/SentEvalRu/'+ kind + '/test.csv', sep='\n', header=None)
        test = DataFrame( test[0].apply(lambda x: re.findall(r'(.*)\t(\d+$)', x)[0] ).tolist() )

        table_len = (train.shape[0], test.shape[0])

        train = train.drop_duplicates()
        test = test.drop_duplicates()

        print('Dublicates removed (train):', table_len[0] - train.shape[0], 'from', table_len[0])
        print('Dublicates removed (test):', table_len[1] - test.shape[0], 'from', table_len[1])

        print('NaN fields in label', train[1].isna().sum() + test[1].isna().sum() )
        print('NaN fields in text', train[0].isna().sum() + test[0].isna().sum() )

        if (len(train[0])!=len(train[1])) or (len(test[0])!=len(test[1])):
          raise LenException

        return train[0], test[0], train[1].astype('int64'), test[1].astype('int64')


      