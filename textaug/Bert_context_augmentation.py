import numpy as np
from keras_bert import load_trained_model_from_checkpoint
import re
import tokenization


class BertContextAugmentation():
    # TODO try catch
    def __init__(self, model_folder):
        self.folder = model_folder
        self.config_path = self.folder + '/bert_config.json'
        self.checkpoint_path = self.folder + '/bert_model.ckpt'
        self.vocab_path = self.folder + '/vocab.txt'
        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_path, do_lower_case=False)
        self.model = load_trained_model_from_checkpoint(self.config_path, self.checkpoint_path, training=True)

    def bert_aug(self, cls_sentence):
        # предсказание слов, закрытых токеном MASK в фразе. На вход нейросети надо подать фразу в формате: [CLS] Я пришел в [MASK] и купил [MASK]. [SEP]

        # входная фраза с закрытыми словами с помощью [MASK]
        # sentence = 'Я пришел в [MASK] и купил [MASK].'  #@param {type:"string"}
        out_sentence = cls_sentence
        sentence = cls_sentence

        # преобразование в токены (tokenizer.tokenize() не обрабатывает [CLS], [MASK], поэтому добавим их вручную)
        sentence = sentence.replace(' [MASK] ', '[MASK]');
        sentence = sentence.replace('[MASK] ', '[MASK]');
        sentence = sentence.replace(' [MASK]',
                                    '[MASK]')  # удаляем лишние пробелы. Можно заменить регуляркой "\s?\[MASK\]\s?", но это надо импортить re
        sentence = sentence.split('[MASK]')  # разбиваем строку по маске
        tokens = ['[CLS]']  # фраза всегда должна начинаться на [CLS]
        # обычные строки преобразуем в токены с помощью tokenizer.tokenize(), вставляя между ними [MASK]
        for i in range(len(sentence)):
            if i == 0:
                tokens = tokens + self.tokenizer.tokenize(sentence[i])
            else:
                tokens = tokens + ['[MASK]'] + self.tokenizer.tokenize(sentence[i])
        tokens = tokens + ['[SEP]']  # фраза всегда должна заканчиваться на [SEP]
        # в tokens теперь токены, которые гарантированно по словарю преобразуются в индексы

        # преобразуем в массив индексов, который можно подавать на вход сети, причем число 103 в нем это [MASK]
        token_input = self.tokenizer.convert_tokens_to_ids(tokens)
        # удлиняем до 512 длины
        token_input = token_input + [0] * (512 - len(token_input))

        # создаем маску, заменив все числа 103 на 1, а остальное 0
        mask_input = [0] * 512
        for i in range(len(mask_input)):
            if token_input[i] == 103:
                mask_input[i] = 1
        # print(mask_input)

        # маска фраз (вторая фраза маскируется числом 1, а все остальное числом 0)
        seg_input = [0] * 512

        # конвертируем в numpy в форму (1,) -> (1,512)
        token_input = np.asarray([token_input])
        mask_input = np.asarray([mask_input])
        seg_input = np.asarray([seg_input])

        # пропускаем через нейросеть...
        predicts = self.model.predict([token_input, seg_input, mask_input])[
            0]  # в [0] полная фраза с заполненными предсказанными словами на месте [MASK]
        predicts = np.argmax(predicts, axis=-1)

        # форматируем результат в строку, разделенную пробелами
        predicts = predicts[0][
                   :len(tokens)]  # длиной как исходная фраза (чтобы отсечь случайные выбросы среди нулей дальше)
        out = []
        # добавляем в out только слова в позиции [MASK], которые маскированы цифрой 1 в mask_input
        for i in range(len(mask_input[0])):
            if mask_input[0][i] == 1:  # [0][i], т.к. требование было (1,512)
                out.append(predicts[i])

        out = self.tokenizer.convert_ids_to_tokens(out)  # индексы в токены

        for i in range(len(re.findall('\[MASK\]?', out_sentence))):
            out_sentence = re.sub('\[MASK\]', out[i], out_sentence, 1)

        return out_sentence

    def choose_random_place(self, sentence, sent_length):
        """
        Выбирает рандомные места в предложении куда в дальнейшем вставляет слово
        """
        aug_num = np.random.randint(1, sent_length // 3 + 1)
        splited_sent = sentence.split(' ')
        for i in range(aug_num):
            splited_sent = splited_sent[:];
            splited_sent.insert(np.random.randint(1, sent_length), '[MASK]')
        return ' '.join(splited_sent)

    def choose_random_word(self, sentence, sent_length):
        """
        Выберает рандомное слово которое будет заменено
        """
        aug_num = np.random.randint(1, sent_length // 3 + 1)
        splited_sent = sentence.split(' ')
        for i in range(aug_num):
            rand_ind = np.random.randint(0, sent_length)
            for n, i in enumerate(splited_sent):
                if n == rand_ind:
                    splited_sent[n] = '[MASK]'
        return ' '.join(splited_sent)

    def make_single_aug(self, sentence, sent_length):
        """
        Применяет одну из двух аугментаций к предложению
        """
        if np.random.randint(0, 2) == 0:
            aug_sentence = self.bert_aug(self.choose_random_place(sentence, sent_length))
        else:
            aug_sentence = self.bert_aug(self.choose_random_word(sentence, sent_length))
        return aug_sentence

    def try_another_one_aug(self, sentence, sent_length, attempts=1, trys=3):
        """
        Пытается применить аугментацию пока не получит новое предложение.
        """
        if attempts <= trys:
            aug_sentence = self.make_single_aug(sentence, sent_length)
            if aug_sentence == sentence:
                self.try_another_one_aug(sentence, sent_length, attempts=attempts + 1, trys=trys)
            else:
                return aug_sentence
        return sentence

    def make_augmentation(self, sentence, attempts=1, trys=3):
        sent_length = len(sentence.split(' '))
        if sent_length // 3 > 0:
            aug_sentence = self.try_another_one_aug(sentence, sent_length, attempts=attempts, trys=trys)
            return aug_sentence
        return sentence

