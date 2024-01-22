import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from app.data_processing.data_cleaning.training import create_train_test_set
from sklearn import metrics
from sklearn.linear_model import LinearRegression

Stock = pd.read_csv(r"F:\ijse final\ai-module-backend\core\csv\AMZN.csv", index_col=0)
# Rename the column
df_Stock = Stock.rename(columns={"Close(t)": "Close"})

# Drop the 'Date_col' column
df_Stock = df_Stock.drop(columns="Date_col")


def test_function():
    X_train, X_val, X_test, Y_train, Y_val, Y_test = create_train_test_set(df_Stock)
    lr = LinearRegression()
    lr.fit(X_train, Y_train)
    future_date = "2012-12-21"
    future_date = pd.to_datetime(future_date, format="%Y-%m-%d")

    # Prepare features for the future date
    # (Replace this with the actual values of features for the future date)
    future_features = X_train.iloc[0].copy()

    # Reshape features for prediction
    future_features = future_features.values.reshape(1, -1)

    # Predict the stock price for the future date
    future_price = lr.predict(future_features)[0]

    # Print the predicted price for the future date
    print(f"Predicted Price for {future_date}: ${future_price:.2f}")

    # Predictions on the validation set
    Y_val_pred = lr.predict(X_val)

    # Plot the actual and predicted values
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
    plt.show()

    # Your data cleaning code goes here
    print("Data cleaning function executed.")
    return "ok"
