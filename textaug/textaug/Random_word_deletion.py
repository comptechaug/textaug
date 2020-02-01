import random


class RandomWordDeletion:
    # randomly deletes one word from sentence
    def make_augmentation(self, sentence):
        words = sentence.split()
        sentence_length = len(words)

        if sentence_length == 1:
            return sentence

        random_ind = random.randint(0, sentence_length - 1)
        del words[random_ind]
        return ' '.join(words)