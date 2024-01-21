import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load your stock data
Stock = pd.read_csv(r"F:\ijse final\ai-module-backend\core\csv\AMZN.csv", index_col=0)

# Rename the column
df_Stock = Stock.rename(columns={"Close(t)": "Close"})

# Drop the 'Date_col' column
df_Stock = df_Stock.drop(columns="Date_col")


# Assuming 'create_train_test_set' is defined elsewhere in your code
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


# Create train, validation, and test sets
X_train, X_val, X_test, Y_train, Y_val, Y_test = create_train_test_set(df_Stock)

# Train a linear regression model
lr = LinearRegression()
lr.fit(X_train, Y_train)

# Future date for prediction
future_date = "2070-02-21"
future_date = pd.to_datetime(future_date, format="%Y-%m-%d")


# Assuming you have a function to get features for a specific date
def get_features_for_date(df, date):
    # Replace this with the actual logic to get features for the given date
    # For example, you may want to filter the dataframe based on the date
    features_for_date = df[df.index == date].drop(columns=["Close_forcast"], axis=1)
    return features_for_date


# Prepare features for the future date
future_features = get_features_for_date(df_Stock, future_date)

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
    future_date, future_price, color="red", marker="o", label="Predicted Future Price"
)
plt.title("Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(which="major", color="k", linestyle="-.", linewidth=0.5)
plt.show()
