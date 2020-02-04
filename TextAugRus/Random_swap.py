import random


class RandomSwapAug:
    def make_augmentation(self, sentence):
        words = sentence.split(" ")
        if len(words) == 1:
            return sentence
        new_words = words.copy()
        n = random.randint(0, round(len(new_words) / 2) - 1)
        random_idx_1 = random.randint(0, len(new_words) - 1)
        random_idx_2 = random_idx_1
        while random_idx_2 == random_idx_1:
            random_idx_2 = random.randint(0, len(new_words) - 1)
        new_words[random_idx_1], new_words[random_idx_2] = (
            new_words[random_idx_2],
            new_words[random_idx_1],
        )
        str = " ".join(new_words)
        return str
