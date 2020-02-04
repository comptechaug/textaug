from yandex_translate import YandexTranslate


class Translate:
    def __init__(self, yandex_translate_api_key="",  traslate_language='en'):
        self.trans = YandexTranslate(yandex_translate_api_key)
        self.traslate_language =  traslate_language

    def make_augmentation(self, text, batch_size=20):
        if isinstance(text, str):
            return self.trans.translate(
                self.trans.translate(text, "ru-"+self.traslate_language)["text"], self.traslate_language+"-ru"
            )["text"]
        else:
            return [
                self.trans.translate(
                    self.trans.translate(text[i : i + batch_size], "ru-en")["text"],
                    "en-ru",
                )["text"]
                for i in range(0, len(text), batch_size)
            ]
