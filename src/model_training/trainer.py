import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.externals import joblib
from src.text_processing.preprocessor import Trainer
from src.text_processing.preprocessor import tokenize_text
from settings import MODEL_PATH, JOB_TITLE_CSV_PATH


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


if __name__ == '__main__':

    train_model(train_df_path=JOB_TITLE_CSV_PATH)
