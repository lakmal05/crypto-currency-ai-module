import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression


def create_train_test_set(df_Stock):
    features = df_Stock.drop(columns=["Close_forcast"], axis=1)
    target = df_Stock["Close_forcast"]

    data_len = df_Stock.shape[0]
    print("Historical Stock Data length is - ", str(data_len))

    # create a chronological split for train and testing
    train_split = int(data_len * 0.88)
    print("Training Set length - ", str(train_split))

    val_split = train_split + int(data_len * 0.1)
    print("Validation Set length - ", str(int(data_len * 0.1)))

    print("Test Set length - ", str(int(data_len * 0.02)))

    # Splitting features and target into train, validation and test samples
    X_train, X_val, X_test = (
        features[:train_split],
        features[train_split:val_split],
        features[val_split:],
    )
    Y_train, Y_val, Y_test = (
        target[:train_split],
        target[train_split:val_split],
        target[val_split:],
    )

    # print shape of samples
    print(X_train.shape, X_val.shape, X_test.shape)
    print(Y_train.shape, Y_val.shape, Y_test.shape)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test
