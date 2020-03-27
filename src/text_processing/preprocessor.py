import string
import spacy

spacy.load('en_core_web_sm')

from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.base import TransformerMixin
parser = English()


class Trainer(TransformerMixin):

    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self


def get_params():
    return {}


def clean_text(text):

    text = text.strip().replace("\n", " ").replace("\r", " ")
    text = text.lower()

    return text


def tokenize_text(sample):

    symbols = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”"]

    tokens = parser(sample)
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)

    tokens = lemmas
    tokens = [tok for tok in tokens if tok not in STOP_WORDS]
    tokens = [tok for tok in tokens if tok not in symbols]

    return tokens


if __name__ == '__main__':

    tokenize_text(sample="")
