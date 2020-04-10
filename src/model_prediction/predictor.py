import pandas as pd
import time

from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from joblib import load
from src.model_training.trainer import get_bow_vector
from settings import MODEL_PATH, PREDICT_CSV_PATH

print("load model...")
st = time.time()
model = load(MODEL_PATH)
ed = time.time()
print(ed - st)


def predict_job_tag(model_path, df_path):

    pipe = joblib.load(model_path)
    df = pd.read_csv(df_path)
    test_x = df['Title'].tolist()
    # test_df
    predictions = pipe.predict(test_x)
    new_df = pd.DataFrame(list(zip(test_x, predictions)), columns=["Title", "Tag"])
    new_df.to_csv(df_path, index=True, header=True, mode="w")

    return


def predict_df(model_path, test_df_path):

    pipe = joblib.load(model_path)
    test_df = pd.read_csv(test_df_path)
    test_x = test_df['Title_Input'].tolist()
    test_labels = test_df['Tag_Output'].tolist()
    # test_df
    predictions = pipe.predict(test_x)
    accuracy = accuracy_score(test_labels, predictions)

    return accuracy


def predict_tags(titles):

    test_bow_vectors = get_bow_vector(df_titles=titles)
    predictions = model.predict(test_bow_vectors)

    return predictions


def estimate_accuracy(test_df_path):

    test_df = pd.read_csv(test_df_path)
    test_titles = test_df["Title"].values.tolist()
    predicted_tags = predict_tags(titles=test_titles)
    test_tags = test_df["Tag"].values.tolist()
    accuracy = accuracy_score(test_tags, predicted_tags)

    return accuracy


def predict_new_tags(new_df_path):

    new_df = pd.read_csv(new_df_path)
    new_titles = new_df["Title"].values.tolist()
    new_predictions = predict_tags(titles=new_titles)
    new_df = pd.DataFrame(list(zip(new_titles, new_predictions)), columns=["Title", "Tag"])
    new_df.to_csv(PREDICT_CSV_PATH, index=True, header=True, mode="w")

    return PREDICT_CSV_PATH


if __name__ == '__main__':

    predict_job_tag(model_path=MODEL_PATH, df_path="")
