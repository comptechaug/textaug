import random


class RandomSwapAug:
    def make_augmentation(self, sentence):
        words = sentence.split(' ')
        new_words = words.copy()
        n = random.randint(0, round(len(new_words) / 2) - 1)
        for _ in range(n):
            random_idx_1 = random.randint(0, len(new_words) - 1)
            random_idx_2 = random_idx_1
            counter = 0
            while random_idx_2 == random_idx_1:
                random_idx_2 = random.randint(0, len(new_words) - 1)
                counter += 1
                if counter > 3:
                    str = ' '.join(new_words)
                    return str
            new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
            str = ' '.join(new_words)
            return str
