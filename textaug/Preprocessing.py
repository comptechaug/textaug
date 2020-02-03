from nltk.stem.snowball import SnowballStemmer
from pymystem3 import Mystem
import pymorphy2
import re


class TextPreprocessor(object):
    def __init__(self, lemmatizer="stemmer_lemmatisation", lemmatization=False):
        self.lemmatization = lemmatization
        self.lemmatizer = lemmatizer
        self.stemmer = SnowballStemmer("russian")
        self.morph = pymorphy2.MorphAnalyzer()
        self.lemmatizers = {
            "stemmer_lemmatisation": self.stemmer_lemmatisation,
            "pymorphy2_lemmatisation": self.pymorphy2_lemmatisation,
            "mystem_lemmatisation": self.mystem_lemmatisation,
        }

    def pymorphy2_lemmatisation(self, text):
        return " ".join(
            [self.morph.parse(word)[0].normal_form for word in text.split()]
        )

    def stemmer_lemmatisation(self, text):
        return " ".join([self.stemmer.stem(word) for word in text.split()])

    def mystem_lemmatisation(self, text):
        return "".join(Mystem().lemmatize(text))

    def standardize_text(self, text):
        text = re.sub(r"http\S+", r"", text)
        text = re.sub(r"[\W\d]", r" ", text)
        text = re.sub(r"(\w)+\1", r"", text)
        #     text = re.sub(r'((.+)\2)', r'', text)
        text = re.sub(r"_", r"", text)
        text = text.lower()
        return text

    def text_preparation(self, text):
        new_text = self.standardize_text(text)
        if new_text.split():
            if self.lemmatization:
                return self.lemmatizers[self.lemmatizer](new_text)
            else:
                return new_text
        return None
