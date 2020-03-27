from utils.train_test_creator import create_train_test_data
from src.model_training.trainer import train_model
from src.model_prediction.predictor import predict_df
from settings import TEST_RATIO


if __name__ == '__main__':

    train_path, test_path = create_train_test_data(test_ratio=TEST_RATIO)
    model_path = train_model(train_df_path=train_path)
    accuracy = predict_df(model_path=model_path, test_df_path=test_path)
    print("Accuracy:", accuracy)
