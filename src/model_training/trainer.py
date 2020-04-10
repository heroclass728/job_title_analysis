import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.externals import joblib
from joblib import dump
from src.corpus.creator import Trainer
from src.corpus.creator import tokenize_text
from settings import MODEL_PATH, CORPUS_PATH

print("load corpus...")
f = open(CORPUS_PATH, 'r')
corpus = eval(f.read())
f.close()
corpus_keys = list(corpus.keys())


def train_model(train_df_path):

    train = pd.read_csv(train_df_path)
    df_vectorization = CountVectorizer(tokenizer=tokenize_text, ngram_range=(1, 5))
    clf = SVC(kernel='linear')

    pipe = Pipeline([('cleanText', Trainer()), ('vectorizer', df_vectorization), ('clf', clf)])
    # data
    train_df = train['Title_Input'].tolist()
    train_labels = train['Tag_Output'].tolist()

    pipe.fit(train_df, train_labels)
    joblib.dump(pipe, MODEL_PATH)
    del pipe

    return MODEL_PATH


def get_bow_vector(df_titles):

    title_vectors = []
    for title in df_titles:
        sentence_tokens = tokenize_text(sample=title)
        sent_vec = [0] * len(corpus)

        for token in sentence_tokens:
            if token in corpus_keys:
                index = corpus_keys.index(token)
                sent_vec[index] = 1

        title_vectors.append(sent_vec)

    return title_vectors


def train_model_bow(train_df_path):

    train_df = pd.read_excel(train_df_path)
    train_titles = train_df["Title"].values.tolist()
    train_tags = train_df["Tag"].values.tolist()

    bow_vector = get_bow_vector(df_titles=train_titles)

    training_model = SVC(kernel='rbf')
    training_model.fit(bow_vector, train_tags)
    dump(training_model, MODEL_PATH)

    return MODEL_PATH


if __name__ == '__main__':

    train_model_bow(train_df_path="")
