# TextAugRus
Library for russian text augmentation
CompTech Winter School 2020

Предложения и вопросы пишите на почту *comptechaugteam@gmail.com*

# Requirements
  Python 3.7.3

  nltk 3.4.5

  pymystem3 0.2.0

  pymorphy2 0.8

  tensorflow 1.15.0

  tensorflow-gpu

  numpy 1.18.1

  keras-bert 0.81.0

  yandex.translate 0.3.5

  gensim 3.8.1

  annoy 1.16.3

  six 1.14.0

  pandas 1.0.0

  keras 2.3.1

  tokenization 1.0.7
 
# Installation
`pip install TextAugRus`

# Available augmentations
  * **Random word deletion**
    * Class name: WordDeletion
    Arguments: share_of_changes - percent of words to be deleted (default = 0.3)
    
  * **Word deletion with tf-idf rating** - use td-idf values as criteria to delete words (deletes words with the smallest rates)
    * Class name: WordDeletion
    Arguments: share_of_changes - percent of words to be deleted (default = 0.3) 

  * **Random swap augmentation**
    * Class name: RandomSwapAug

  * **Translate augmentation**
    * Class name: Translate
    Arguments: share_of_changes - percent of words to be deleted (default = 0.3) 

  * **Word2vec augmentation**
    * Class name: Word2VecAugmentation
    Arguments: path_to_dictionary (default = "mipt_vecs.w2v"), indexer (default = None), cache_dict (default = False), partition (default = 0.5)

  * **Bert context augmentation**
    * Class name: BertContextAugmentation
    Arguments: model_folder
  
# Example use cases:
``` python
from TextAugRus.Word2vec_augmentation import Word2VecAugmentation
w2v_aug = Word2VecAugmentation()
sentence = 'Какое-то русское предложение'
sentence = w2v_aug(make_augmentation(sentence)
```

# Benchmarks
### Twitter dataset (2 labels) -- balanced accuracy score (%)

[Датасет твитов: задача тонового анализа](http://study.mokoron.com/) 

|Embeddings | TF-IDF |  | | Fasttext ||
|------------|--------|-------|-------|-----|-------|
|**Methods** | **Log Reg** | **Naive Bayes** | **XGboost** | **Log Reg** | **XGboost** |
|No augmentation | *74.9* | 74.7 | 68.7 | 70.1 | 68 |
|Word2Vec | *74.8* | 74.1 | 66.1 | 69.3 | 68.8 |
|Random word deletion | *75.2* | 74.1 | 70.0 | 70.1 | 68.8 |

### Poems dataset (33 labels) -- weighted F1-score

[Датасет поэм: многоклассовая классификация](https://github.com/comptechml/SentEvalRu/tree/master/data)

|Embeddings | TF-IDF |  | | Fasttext ||
|------------|--------|-------|-------|-----|-------|
|**Methods** | **Log Reg** | **Naive Bayes** | **XGboost** | **Log Reg** | **XGboost** |
|No augmentation | 0.216 | 0.109 | 0.198 | 0.201 | *0.353* |
|Word2Vec | 0.222 | 0.123 | 0.254 | 0.225 | *0.358* |
|Random word deletion | 0.221 | 0.129 | 0.282 | 0.237 | *0.325* |
|Translation (ru->hi->ru) | 0.211 | 0.141 | 0.251 | 0.219 | *0.356* |
|Bert | 0.217 | 0.133 | 0.230 | 0.237 | *0.351* |
|All augmentations | 0.236 | 0.156 | 0.251 | 0.253 | *0.344* |

# References

[Частотный словарь русского языка](http://dict.ruslang.ru/freq.php)

[Alexander Panchenko](http://panchenko.me/) : [Модель Word2Vec](http://panchenko.me/data/dsl-backup/w2v-ru/all.norm-sz100-w10-cb0-it1-min100.w2v)

[DeepPavlov](http://files.deeppavlov.ai) : [Модель BERT](http://files.deeppavlov.ai/deeppavlov_data/bert/rubert_cased_L-12_H-768_A-12_v2.tar.gz)

