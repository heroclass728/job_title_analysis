import re
import nltk

nltk.download('stopwords')
nltk.download('punkt')
import string
import pandas as pd

from nltk.corpus import stopwords
from settings import TRAIN_DATA_CSV_PATH, TEST_DATA_CSV_PATH, JOB_TITLE_CSV_PATH


def get_first_title(title):
    # keep "co-founder, co-ceo, etc"
    title = re.sub(r"[Cc]o[\- ]", "", title)
    split_titles = re.split(r",|-|\||&|:|/|and", title)
    return split_titles[0].strip()


def get_title_features(title):
    features = {}
    word_tokens = nltk.word_tokenize(title)
    filtered_words = [w for w in word_tokens if w not in stop_words]
    filtered_tokens = [tok for tok in filtered_words if tok not in symbols]
    for word in filtered_tokens:
        features['contains({})'.format(word.lower())] = True
    if len(filtered_tokens) > 0:
        first_key = 'first({})'.format(filtered_tokens[0].lower())
        last_key = 'last({})'.format(filtered_tokens[-1].lower())
        features[first_key] = True
        features[last_key] = True
    return features


def create_department_features(title_df):

    raw_job_titles = []
    titles = title_df['Title'].values.tolist()
    tags = title_df['Tag'].values.tolist()

    for title, tag in zip(titles, tags):
        tmp_dict = {"title": title, "department": tag}
        raw_job_titles.append(tmp_dict)

    departments_features = [
        (
            get_title_features(job_title["title"]),
            job_title["department"]
        )
        for job_title in raw_job_titles
        if job_title["department"] is not None
    ]

    return departments_features


if __name__ == '__main__':

    stop_words = set(stopwords.words('english'))
    symbols = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”"]
    job_df = pd.read_csv(JOB_TITLE_CSV_PATH)
    train_df = pd.read_csv(TRAIN_DATA_CSV_PATH)
    test_df = pd.read_csv(TEST_DATA_CSV_PATH)

    job_department_features = create_department_features(title_df=job_df)
    train_department_features = create_department_features(title_df=train_df)
    test_department_features = create_department_features(title_df=test_df)

    # d_size = int(len(departments_features) * 0.5)
    # d_train_set = departments_features[d_size:]
    # d_test_set = departments_features[:d_size]
    departments_classifier = nltk.NaiveBayesClassifier.train(
        job_department_features
    )
    print("Department classification accuracy: {}".format(
        nltk.classify.accuracy(
            departments_classifier,
            test_department_features
        )
    ))
