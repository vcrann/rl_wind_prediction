import tensorflow as tf
import keras
from keras import layers
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

# expected input data shape: (batch, timesteps, data_dim)
data_dim = 6
model = keras.Sequential()
model.add(keras.Input(shape=(None, data_dim)))
# model.add(layers.LSTM(420, return_sequences=True, input_shape=(None, data_dim)))
model.add(layers.LSTM(420, return_sequences=True))
model.add(layers.LSTM(180, return_sequences=True))
model.add(
    layers.LSTM(90, return_sequences=True, dropout=0.96)
)  # Check this droput, seems high?
model.add(layers.LSTM(48, activation="relu"))
model.add(layers.Dense(2))
# model.compile(optimizer="adam", loss="mse", metrics=["accuracy"], learning_rate=4e-5)
optimizer = keras.optimizers.RMSprop(learning_rate=4e-5)
model.compile(optimizer=optimizer, loss="mse", metrics=["accuracy"])
# print(model.summary())

df = pd.read_csv("data/training_data.csv")

uas_state_data = df.iloc[:, 3:]  # X
wind_data = df.iloc[:, 1:3]  # Y

ss = StandardScaler()
mm = MinMaxScaler()

state_ss = ss.fit_transform(uas_state_data)
wind_data_mm = mm.fit_transform(wind_data)

state_train = state_ss[:1800, :]
state_test = state_ss[1800:, :]

wind_train = wind_data_mm[:1800, :]
wind_test = wind_data_mm[1800:, :]

# Line up state with wind 1 timestep ahead
state_train = state_train[:-1]
state_test = state_test[:-1]

wind_train = wind_train[1:]
wind_test = wind_test[1:]

# state_train = np.reshape(state_train, (int(batch), 50, 6))

state_train = np.expand_dims(state_train, 1)
state_test = np.expand_dims(state_test, 1)
print(state_train.shape)

training = model.fit(
    state_train,
    wind_train,
    epochs=1500,
    batch_size=16,
    verbose=2,
    validation_data=(state_test, wind_test),
)

plt.plot(training.history["loss"])
plt.plot(training.history["val_loss"])
plt.title("train vs validation loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "validation"], loc="upper right", ncol=2)

plt.show()

model.save("models/keras/model_keras_3.keras")
