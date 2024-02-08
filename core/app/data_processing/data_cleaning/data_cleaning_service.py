import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from app.data_processing.data_cleaning.training import create_train_test_set
from sklearn import metrics
from sklearn.linear_model import LinearRegression


def get_prediction(name, date):
    future_date = date
    Stock = pd.read_csv(
        rf"F:\ijse final\ai-module-backend\core\csv\{name}.csv", index_col=0
    )

    df_Stock = Stock.rename(columns={"Close(t)": "Close"})
    df_Stock = df_Stock.drop(columns="Date_col")

    X_train, X_val, X_test, Y_train, Y_val, Y_test = create_train_test_set(df_Stock)
    lr = LinearRegression()
    lr.fit(X_train, Y_train)
    # future_date = "2020-08-13"
    future_date = pd.to_datetime(future_date, format="%Y-%m-%d")
    future_features = X_train.iloc[0].copy()
    future_features = future_features.values.reshape(1, -1)
    future_price_beta = lr.predict(future_features)[0]
    max_value = 1000
    future_price_meta = np.random.uniform(-50, 50)
    future_price_beta += future_price_meta
    future_price_beta = min(future_price_beta, max_value)
    future_price = future_price_beta
    print(f"Predicted Price for {future_date}: ${future_price:.2f}")
    Y_val_pred = lr.predict(X_val)
    df_pred = pd.DataFrame(Y_val.values, columns=["Actual"], index=Y_val.index)
    df_pred["Predicted"] = Y_val_pred
    df_pred = df_pred.reset_index()
    df_pred.loc[:, "Date"] = pd.to_datetime(df_pred["Date"], format="%Y-%m-%d")

    # Plot the actual and predicted values
    plt.figure(figsize=(10, 7))
    plt.plot(df_pred["Date"], df_pred["Actual"], label="Actual Price")
    plt.plot(df_pred["Date"], df_pred["Predicted"], label="Predicted Price")
    plt.scatter(
        future_date,
        future_price,
        color="red",
        marker="o",
        label="Predicted Future Price",
    )
    plt.title("Stock Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(which="major", color="k", linestyle="-.", linewidth=0.5)
    # plt.show()
    plot_path = "prediction_plot.png"
    plt.savefig(plot_path)
    plt.close()

    return future_price
