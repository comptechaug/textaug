**bench_pipe.py** - файл с бэнчмарком аугментаций для консоли.

## Использование
* введите в терминале `python bench_pipe.py [data folder] [augmentaton]`

`data folder` - путь к папке с данными (там должны находиться файлы train.csv, test.csv)
`augmentation` - название файла (без расширения) с аугментированными трэйновыми данными в папке `data folder` (если аугментированного файла нет, то можно не указывать)

* на выходе в консоле вы получите результат бэнчмарка с применением различных эмбеддингов и классификаторов для ваших данных

## Полученные результаты

### Twitter dataset (2 labels) -- balanced accuracy score (%)

|Embeddings | TF-IDF |  | | Fasttext ||
|------------|--------|-------|-------|-----|-------|
|**Methods** | **Log Reg** | **Naive Bayes** | **XGboost** | **Log Reg** | **XGboost** |
|No augmentation | *74.9* | 74.7 | 68.7 | 70.1 | 68 |
|Word2Vec | *74.8* | 74.1 | 66.1 | 69.3 | 68.8 |
|Random word deletion | *75.2* | 74.1 | 70.0 | 70.1 | 68.8 |

### Poems dataset (33 labels) -- weighted F1-score

|Embeddings | TF-IDF |  | | Fasttext ||
|------------|--------|-------|-------|-----|-------|
|**Methods** | **Log Reg** | **Naive Bayes** | **XGboost** | **Log Reg** | **XGboost** |
|No augmentation | 0.216 | 0.109 | 0.198 | 0.201 | *0.353* |
|Word2Vec | 0.222 | 0.123 | 0.254 | 0.225 | *0.358* |
|Random word deletion | 0.221 | 0.129 | 0.282 | 0.237 | *0.325* |
|Translation\* (ru->hi->ru) | 0.211 | 0.141 | 0.251 | 0.219 | *0.356* |
|Bert | 0.217 | 0.133 | 0.230 | 0.237 | *0.351* |
|All augmentations | 0.236 | 0.156 | 0.251 | 0.253 | *0.344* |

\*: здесь была проаугментирована только 1/3 train набора 
