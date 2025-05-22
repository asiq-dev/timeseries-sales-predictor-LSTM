from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
data = pd.read_csv("MicrosoftStock.csv")

# print(data.head())

# print(data.info())

# print(data.describe())


# Initial data visualization
# plt.figure(figsize=(8,4))
# plt.plot(data['date'], data['open'], label='Open', color='blue')
# plt.plot(data['date'], data['close'], label='Close', color='red')
# plt.title("Open-Close price over time")
# plt.legend()
# plt.show

# # Trading volume check for outliers
# plt.figure(figsize=(8,4))
# plt.plot(data['date'], data['volume'], label="Volume", color='orange')
# plt.title("Stock volume over time")
# plt.show()


# drop non numeric data
# numeric_data = data.select_dtypes(include=['int64', 'float64'])
# print(numeric_data, '&'*30)


#check for correlation between features
# plt.figure(figsize=(6,3))
# sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
# plt.title("Feature correlation Heatmap")
# plt.show()


#Prepare for lstm model sequential
stock_close = data.filter(['close'])
dataset = stock_close.values
training_data_len = int(np.ceil(len(dataset) * 0.95))


#preprocessing stage
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset)
training_data = scaled_data[:training_data_len] #95% data of all data

X_train, y_train = [], []


#create a sliding window for our stock
for i in range(60, len(training_data)):
    X_train.append(training_data[i-60:i, 0])
    y_train.append(training_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


#Build the model
model = keras.models.Sequential()

#first layer
model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1],1)))

#second layer 
model.add(keras.layers.LSTM(64, return_sequences=False))

#third layer(Dense)
model.add(keras.layers.Dense(128, activation='relu'))

#4th layer
model.add(keras.layers.Dropout(0.5))

#final layer
model.add(keras.layers.Dense(1))
model.summary()
model.compile(optimizer="adam",
              loss='mae',
              metrics=[keras.metrics.RootMeanSquaredError()])
traing = model.fit(X_train, y_train, epochs=20, batch_size=32)


# Prepare the test data
test_data = scaled_data[training_data_len - 60:]
X_test, y_test = [], dataset[training_data_len:]


for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


#make prediction 
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Ploting data
train = data[:training_data_len]
test = data[training_data_len:]
test = test.copy()
test['Predictions'] = predictions


plt.figure(figsize=(12,6))
plt.plot(train['date'], train['close'], label="Train(actual)", color='blue')
plt.plot(train['date'], train['close'], label="Test(actual)", color='red')
plt.plot(test['date'], test['Predictions'], label="Predictions", color='green')
plt.title("Our Stock Predictions")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()