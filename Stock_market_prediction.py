#V3
import tkinter as tk
from tkinter import messagebox
import math
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import mplcursors

plt.style.use('seaborn-deep')

# Main Window
window = tk.Tk()
window.title("Stock Price Prediction")
window.geometry("400x300")

# Function to handle button click event
def predict_stock_price():
    stock_symbol = symbol_entry.get()
    num_epochs = int(epochs_entry.get())

    # Fetch stock data for the specified company from 2016-01-01 to 2023-05-15
    stock_data = yf.download(stock_symbol, start='2016-01-01', end='2023-05-15')

    # Create a dataframe using the 'Close' column
    data = stock_data.filter(['Close'])
    # Convert to numpy array
    dataset = data.values
    # Get the number of rows to train the model (90% of the original data)
    training_data_len = math.ceil(len(dataset) * 0.9)
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    # Data preprocessing
    # Create training data set
    train_data = scaled_data[0:training_data_len, :]
    # Split data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(50, len(train_data)):
        x_train.append(train_data[i - 50:i, 0])
        y_train.append(train_data[i, 0])

    # Convert x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Train the model
    history = model.fit(x_train, y_train, batch_size=1, epochs=num_epochs)

    # Create the testing data set
    test_data = scaled_data[training_data_len - 50:, :]
    x_test = []
    y_test = dataset[training_data_len:, :]

    for i in range(50, len(test_data)):
        x_test.append(test_data[i - 50:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    # Get the model's predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    # Unscale the predictions

    #evaluate the model, by getting the root mean squared error(RMSE)
    rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
    percentage_accuracy = 100 - (rmse / np.mean(y_test)) * 100

    # Plot the Loss vs Epoch graph
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'])
    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #cursor
    cursor = mplcursors.cursor(hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(f"{sel.target[0]:.2f}, {sel.target[1]:.2f}"))
    plt.show()

    # Plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    # Visualize the data
    plt.figure(figsize=(20, 10))
    plt.title(f'Stock Market Prices For {stock_symbol}, #{num_epochs} Epoch(s)')
    plt.xlabel('Date (yyyy-mm-dd)', fontsize=14)
    plt.ylabel(f'Closing Price ($USD)', fontsize=14)
    plt.plot(train['Close'], color="black")
    plt.plot(valid[['Predictions']], color="green")
    plt.plot(valid.index, valid['Close'], color="blue")
    plt.legend(['Train', 'Predictions', 'Actual'], loc='lower right')
    plt.show()
    messagebox.showinfo(f"Percentage accuracy : ", str(round(percentage_accuracy, 2)))
    # Try to predict closing price for the next 5 days after 2023-05-15
    sp_quote = yf.download(stock_symbol, start='2016-01-01', end='2023-05-15')
    new_df = sp_quote.filter(['Close'])
    num_days = int(days_entry.get())
    predicted_prices = []

    for i in range(num_days):
        last_50_days = new_df[-50:].values
        last_50_days_scaled = scaler.transform(last_50_days)
        X_test = np.array([last_50_days_scaled])
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        pred_price = model.predict(X_test)
        pred_price = scaler.inverse_transform(pred_price)
        predicted_prices.append(pred_price[0][0])
        last_50_days = np.append(last_50_days, pred_price, axis=0)
        new_df.loc[new_df.index.max() + pd.Timedelta(days=1)] = pred_price[0][0]

    # Display the predicted prices
    messagebox.showinfo(f"Predicted Prices for {num_days} days after 2023-05-15 ($USD)", str(predicted_prices))

# Widgets
title_label = tk.Label(window, text ="Stock Price prediction")
title_label.pack()

symbol_label = tk.Label(window, text="Stock Symbol:")
symbol_label.pack()
symbol_entry = tk.Entry(window)
symbol_entry.pack()

epochs_label = tk.Label(window, text="Number of Epochs:")
epochs_label.pack()
epochs_entry = tk.Entry(window)
epochs_entry.pack()

days_label = tk.Label(window, text="Number of days after 2023-05-15,\nyou want to predict:")
days_label.pack()
days_entry = tk.Entry(window)
days_entry.pack()

predict_button = tk.Button(window, text="Predict", command=predict_stock_price)
predict_button.pack()

# Start the main event loop
window.mainloop()