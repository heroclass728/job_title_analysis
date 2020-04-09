import pandas as pd

from settings import JOB_TITLE_CSV_PATH, TRAIN_DATA_CSV_PATH, TEST_DATA_CSV_PATH
from sklearn.model_selection import train_test_split


def create_train_test_data(test_ratio):

    total_df = pd.read_excel(JOB_TITLE_CSV_PATH)
    total_categories = total_df["Tag"].values.tolist()
    categories = []
    for total_cat in total_categories:
        if total_cat not in categories:
            categories.append(total_cat)

    train_df = pd.DataFrame()
    test_df = pd.DataFrame()

    for cat in categories:

        cat_df = total_df.loc[total_df["Tag"] == cat]
        sub_train, sub_test = train_test_split(cat_df, test_size=test_ratio, random_state=42)
        train_df = pd.concat([train_df, sub_train])
        test_df = pd.concat([test_df, sub_test])

    train_df.to_csv(TRAIN_DATA_CSV_PATH, index=False, mode='w')
    test_df.to_csv(TEST_DATA_CSV_PATH, index=False, mode='w')

    return TRAIN_DATA_CSV_PATH, TEST_DATA_CSV_PATH


if __name__ == '__main__':

    create_train_test_data(test_ratio=0.3)
