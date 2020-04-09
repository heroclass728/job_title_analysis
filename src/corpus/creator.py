import string
import spacy
import pandas as pd
import nltk
spacy.load('en_core_web_sm')
nltk.download('stopwords')
nltk.download('punkt')

from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.base import TransformerMixin
from settings import CORPUS_PATH
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

    sample = sample.replace("/", " ").replace(".", " ")

    symbols = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”", "/"]

    tokens = parser(sample)
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)

    tokens = lemmas
    tokens = [tok for tok in tokens if tok not in STOP_WORDS]
    tokens = [tok for tok in tokens if tok not in symbols]

    return tokens


def create_corpus(corpus_df_path):

    job_df = pd.read_excel(corpus_df_path)
    titles = job_df["Title"].values.tolist()
    word_freq = {}
    for title in titles:

        filtered_tokens = tokenize_text(sample=title)
        for token in filtered_tokens:
            if token == "":
                continue
            if not token.isalpha():
                continue
            if token not in word_freq.keys():
                word_freq[token] = 1
            else:
                word_freq[token] += 1

    f = open(CORPUS_PATH, 'w')
    f.write(str(word_freq))
    f.close()

    return CORPUS_PATH


if __name__ == '__main__':

    create_corpus(corpus_df_path="")
