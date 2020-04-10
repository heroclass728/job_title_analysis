import argparse

from utils.train_test_creator import create_train_test_data
from src.model_training.trainer import train_model_bow
from src.model_prediction.predictor import predict_new_tags, estimate_accuracy, predict_tags
from settings import TEST_RATIO, PREDICTION_ONLY, NEW_PREDICTION_PATH, JOB_TITLE_CSV_PATH, PREDICTION_ONE_TITLE
from src.corpus.creator import create_corpus


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--job_title', action='store', help='A simple job title string')
    job_title = parser.parse_args().job_title

    if PREDICTION_ONLY:

        if PREDICTION_ONE_TITLE:
            job_tag = predict_tags(titles=[job_title])
            print(job_tag[0])
            result = {'department': job_tag}

            return result

        else:

            print("Predicting Titles...")
            saved_path = predict_new_tags(new_df_path=NEW_PREDICTION_PATH)
            print("Successfully predict the new tags and Save it in {}".format(saved_path))

    else:
        print("Dividing the data into train and test...")
        train_path, test_path = create_train_test_data(test_ratio=TEST_RATIO)

        print("Creating new corpus...")
        corpus_path = create_corpus(corpus_df_path=JOB_TITLE_CSV_PATH)
        print("Corpus is saved in {}".format(corpus_path))

        print("Training model...")
        model_path = train_model_bow(train_df_path=JOB_TITLE_CSV_PATH)
        print("Trained model is saved in {}".format(model_path))

        print("Estimating model...")
        accuracy = estimate_accuracy(test_df_path=test_path)
        print("Accuracy:", accuracy)


if __name__ == '__main__':

    main()
