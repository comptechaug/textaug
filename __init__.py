import os
from .translate import TranslateAug
from .word2vec import Word2vec
from .bert import BertAug
from .synonym import SynonymAug
from .worddrop import WordDropAug
from .wordswap import WordSwapAug

name = "textaug"

__version__ = '0.1'
__author__ = 'CompTechAugTeam'

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

__all__ = [
    'BertAug',
    'Word2vec',
    'SynonymAug',
    'WordDropAug',
    'TranslateAug',
    'WordSwapAug'
]
