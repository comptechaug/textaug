import re
import gensim.models
from gensim.similarities.index import AnnoyIndexer
import random

class Word2VecAugmentation:
    def __init__(self, path_to_dictionary='mipt_vecs.w2v', indexer=None, cache_dict=False):
        #если нет словаря, то скачать его
        self.model = gensim.models.KeyedVectors.load_word2vec_format(path_to_dictionary, binary=True, unicode_errors='ignore')
        self.annoy_index = AnnoyIndexer(self.model, num_trees=10) if (indexer=='annoy') else None
        self.replace_dict = dict() if cache_dict else None

    def clear_replace_dict(self):
        if self.replace_dict is not None:
            self.replace_dict.clear()
        return

    def get_similar_words_dict(self, words_set):
        word_index = 0 if self.annoy_index is None else 1
        replace_dict = dict() if self.replace_dict is None else self.replace_dict
        for word in ( words_set - replace_dict.keys() ):
            if word in self.model:
                replace_dict[word] = self.model.most_similar(word, topn=word_index+1, indexer=self.annoy_index)[word_index][0]
        return replace_dict

    def make_augmentation(self, text, partition=0.5):
        words_set = set( re.findall(r'[а-яА-Яa-zA-Z]+', text) )
        replace_dict = self.get_similar_words_dict(words_set)
        return re.sub(r'[а-яА-Яa-zA-Z]+', lambda x: replace_dict[x.group()] if (random.random()<partition) and (x.group() in replace_dict) else x.group(), text)
