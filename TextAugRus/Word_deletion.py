import random
# import pandas as pd
import pymorphy2


class WordDeletion(object):
    def __init__(
        self,
        use_tf_idf_rating=False,
        share_of_changes=0.3,
        path_to_dictionary="../dictionaries/freqrnc2011.csv",
    ):
        self.use_tf_idf_rating = use_tf_idf_rating
        self.share_of_changes = share_of_changes
        if self.use_tf_idf_rating:
            self.df_dict = pd.read_csv(path_to_dictionary, sep="\t", header=None)
            self.morph = pymorphy2.MorphAnalyzer()
            self.df_dict.drop(0, inplace=True)
            self.df_dict[2] = self.df_dict[2].astype("float64")
            self.dict_median = self.df_dict[2].median(axis=0)

    # randomly deletes one word from sentence

    def tf_idf_rating(self, sentence):
        tf_dict = {}
        words = sentence.split()
        sentence_length = len(words)
        if sentence_length == 1:
            return sentence
        for word in words:
            tf_dict[word] = words.count(word)
        for word, counter in tf_dict.items():
            if self.df_dict[self.df_dict[0] == self.morph.parse(word)[0].normal_form][
                2
            ].empty:
                tf_dict[word] = counter / self.dict_median
            else:
                tf_dict[word] = (
                    counter
                    / self.df_dict[
                        self.df_dict[0] == self.morph.parse(word)[0].normal_form
                    ][2].mean()
                )
        sort_list = sorted(list(tf_dict.items()), key=lambda i: i[1])
        for i in range(int(sentence_length * self.share_of_changes)):
            words.remove(sort_list[i][0])
        return " ".join(words)

    def delete_random_word(self, sentence):
        print("Here")
        words = sentence.split()
        sentence_length = len(words)
        if sentence_length == 1:
            return sentence
        # how many words to delete
        how_many = random.randint(1, int(self.share_of_changes * sentence_length))

        # choose a random word in the sentence and delete it, how_many times
        for i in range(how_many):
            to_delete = random.randint(0, sentence_length - 1)
            del words[to_delete]
            sentence_length -= 1
        return " ".join(words)

    def make_augmentation(self, sentence):
        if self.use_tf_idf_rating:
            return self.tf_idf_rating(sentence)
        else:
            return self.delete_random_word(sentence)

