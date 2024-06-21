import pandas as pd
import os

claseses = os.listdir("dataset/Training")
claseses.sort()

df_train = pd.DataFrame(columns=["file_path", "label"])
df_test = pd.DataFrame(columns=["file_path", "label"])

for idx, c in enumerate(claseses):

    train_files = os.listdir("dataset/Training/" + c)
    test_files = os.listdir("dataset/Testing/" + c)

    train_files = ["data/dataset/Training/" + c + "/" + train_file for train_file in train_files]
    test_files = ["data/dataset/Testing/" + c + "/" + test_file for test_file in test_files]

    for train in train_files:
        df_train_app = {'file_path': train, 'label' : idx}
        df_train.loc[len(df_train)] = df_train_app

    for test in test_files:
        df_test_app = {'file_path': test, 'label' : idx}
        df_test.loc[len(df_test)] = df_test_app

df_train.to_csv("train.csv")
df_test.to_csv("test.csv")
