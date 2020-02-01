from nltk.stem.snowball import SnowballStemmer
from pymystem3 import Mystem
import pymorphy2
import re


class TextPreprocessor(object):
    def __init__(self):
        self.stemmer = SnowballStemmer("russian")
        self.morph = pymorphy2.MorphAnalyzer()
        self.lemmatizers =  {'stemmer_lemmatisation' : self.stemmer_lemmatisation,
                             'pymorphy2_lemmatisation' : self.pymorphy2_lemmatisation,
                             'mystem_lemmatisation' : self.mystem_lemmatisation,}

    def pymorphy2_lemmatisation(self, text):
        new_text = ""
        for word in[self.stemmer.stem(word) for word in text.split()]:
            new_text += self.morph.parse(word)[0].normal_form + " "
        return (new_text)
    def stemmer_lemmatisation(self, text):
        new_text = ""
        for word in[self.stemmer.stem(word) for word in text.split()]:
              new_text += word + " "
        return (new_text)

    def mystem_lemmatisation(self, text):
        lemmas = Mystem().lemmatize(text)
        return (''.join(lemmas))

    def standardize_text(self, text):
        text = re.sub(r"http\S+", r"", text)
        text = re.sub(r'[\W\d]', r' ', text)
        text = re.sub(r'(\w)+\1', r'', text)
    #     text = re.sub(r'((.+)\2)', r'', text)
        text = re.sub(r'_', r'', text)
        text = text.lower()
        return text

    def text_preparation(self, text, lemmatizer='stemmer_lemmatisation', lemmatization = False):

        new_text = self.standardize_text(text)
        if new_text.split():
            if lemmatization:
                return self.lemmatizers[lemmatizer](new_text)
            else:
                return new_text
        return None
