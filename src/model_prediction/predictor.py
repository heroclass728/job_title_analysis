import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from settings import MODEL_PATH, TEST_DATA_CSV_PATH


def predict_df(model_path, test_df_path):

    pipe = joblib.load(model_path)
    test_df = pd.read_csv(test_df_path)
    test_x = test_df['Title_Input'].tolist()
    test_labels = test_df['Tag_Output'].tolist()
    # test_df
    predictions = pipe.predict(test_x)
    accuracy = accuracy_score(test_labels, predictions)

    return accuracy


if __name__ == '__main__':

    predict_df(model_path=MODEL_PATH, test_df_path=TEST_DATA_CSV_PATH)
