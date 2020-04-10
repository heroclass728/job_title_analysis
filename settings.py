import os

from utils.folder_file_manager import make_directory_if_not_exists


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
JOB_TITLE_CSV_PATH = os.path.join(CUR_DIR, 'job_titles.csv')
TRAIN_DATA_CSV_PATH = os.path.join(CUR_DIR, 'train.csv')
TEST_DATA_CSV_PATH = os.path.join(CUR_DIR, 'test.csv')
MODEL_DIR_PATH = make_directory_if_not_exists(os.path.join(CUR_DIR, 'utils', 'model'))
MODEL_PATH = os.path.join(MODEL_DIR_PATH, "job_title_model.joblib")
CORPUS_PATH = os.path.join(MODEL_DIR_PATH, 'corpus.txt')
PREDICT_CSV_PATH = os.path.join(CUR_DIR, 'predict.csv')

TEST_RATIO = 0.3
PREDICTION_ONLY = True
PREDICTION_ONE_TITLE = True

NEW_PREDICTION_PATH = ""
