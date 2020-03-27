from utils.train_test_creator import create_train_test_data
from src.model_training.trainer import train_model
from src.model_prediction.predictor import predict_df
from settings import TEST_RATIO, PREDICTION_ONLY, MODEL_PATH


if __name__ == '__main__':

    train_path, test_path = create_train_test_data(test_ratio=TEST_RATIO)
    if PREDICTION_ONLY:
        print("Predicting model...")

        accuracy = predict_df(model_path=MODEL_PATH, test_df_path=test_path)
        print("Accuracy:", accuracy)
    else:
        print("Training model...")
        model_path = train_model(train_df_path=train_path)
        print("Predicting model...")
        accuracy = predict_df(model_path=MODEL_PATH, test_df_path=test_path)
        print("Accuracy:", accuracy)
