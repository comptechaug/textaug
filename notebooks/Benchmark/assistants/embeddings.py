from sklearn.feature_extraction.text import TfidfVectorizer
from pandas import DataFrame
import numpy
import fasttext

class Embeddings:
  def __init__(self, X_train, X_test, y_train, y_test):
      if( (len(X_train) != len(y_train)) or (len(X_test) != len(y_test)) ):
          raise LenException

      self.split_index = len(X_train)
      self.X = X_train.append(X_test, ignore_index=True).copy()
      self.y_train = y_train
      self.y_test = y_test
      self.model_ft = None

  def change_data(self, X_train, X_test, y_train, y_test):
      self.X = X_train.append(X_test, ignore_index=True).copy()
      self.y_train = y_train
      self.y_test = y_test
      return

    ###TF_IDF
  
  def tf_idf_vectorizer(self):
      vectorizer = TfidfVectorizer()
            
      X_vec = vectorizer.fit_transform(self.X)
      if( X_vec[:self.split_index, ].shape[0]!=len(self.y_train)):
          raise LenException
      return ( X_vec[:self.split_index, ], X_vec[self.split_index:, ], self.y_train, self.y_test )

    ###FASTTEXT

  def fasttext_vectorize_words_list(self, model, words_list):
      if len(words_list)==0:
          return numpy.zeros(300)
      return numpy.array( [self.model_ft.get_word_vector(w) for w in words_list] ).mean(axis=0);

  def fasttext_vectorizer(self, fasttext_path='cc.ru.300.bin'):
      if self.model_ft is None:
          self.model_ft = fasttext.load_model(fasttext_path)
      X_ft = self.X.str.findall(r'[а-яА-ЯёЁ]+')
      X_ft = DataFrame( X_ft.apply(lambda x: self.fasttext_vectorize_words_list(self.model_ft, x) ).tolist() )
      if( X_ft.iloc[:self.split_index, ].shape[0]!=len(self.y_train)):
          raise LenException
      return (X_ft.iloc[:self.split_index, ], X_ft.iloc[self.split_index:, ], self.y_train, self.y_test)

  def fasttext_model_clear_from_ram(self):
      if self.model_ft is None:
          return
      del self.model_ft
      self.model_ft = None
      return