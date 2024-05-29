import numpy as np
import random
import keras
from keras import layers
from data_generator import DataGenerator
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

lookback = np.load(
    "data/earth_sciences_anemometer/training_data/worst_day_X_20_1.npy"
)  # Lookback
lookforward = np.load(
    "data/earth_sciences_anemometer/training_data/worst_day_Y_20_1.npy"
)  # Prediction

lookback_window = 20
prediction_window = 1

# Test train split
test_train_split = 0.8
train_split_index = int(lookback.shape[0] * test_train_split)

lookback_train = lookback[:train_split_index]
lookback_validate = lookback[train_split_index:]
lookforward_train = lookforward[:train_split_index]
lookforward_validate = lookforward[train_split_index:]

# Normalise data
mm = (
    MinMaxScaler()
)  # As output is the same as input, we can use the same scaler (I think!)
# TODO try normalising the output data separately

lookback_train_mm = mm.fit_transform(
    lookback_train.reshape(-1, lookback_train.shape[-1])
).reshape(lookback_train.shape)
lookback_validate_mm = mm.transform(
    lookback_validate.reshape(-1, lookback_validate.shape[-1])
).reshape(lookback_validate.shape)

# lookback_train_mm = mm.fit_transform(lookback_train)
# lookback_validate_mm = mm.transform(lookback_validate)
# lookforward_train_mm = mm.transform(lookforward_train)
# lookforward_validate_mm = mm.transform(lookforward_validate)

# Shuffle training data
p = np.random.permutation(lookback_train_mm.shape[0])
lookback_train_mm = lookback_train_mm[p]
# lookforward_train_mm = lookforward_train_mm[p]
lookforward_train = lookforward_train[p]

data_dim = 3
model = keras.Sequential()
model.add(keras.Input(shape=(None, data_dim)))
# model.add(layers.LSTM(420, return_sequences=True, input_shape=(None, data_dim)))
model.add(layers.LSTM(420, return_sequences=True))
model.add(layers.LSTM(180, return_sequences=True))
model.add(
    layers.LSTM(90, return_sequences=True, dropout=0.96)
)  # Check this droput, seems high?
model.add(layers.LSTM(48, activation="relu"))  # changed from 48
model.add(layers.Dense(data_dim * prediction_window))
model.add(
    layers.Reshape((prediction_window, data_dim))
)  # TODO check if this affects training speed significantly
# model.compile(optimizer="adam", loss="mse", metrics=["accuracy"], learning_rate=4e-5)
optimizer = keras.optimizers.RMSprop(learning_rate=4e-5)
model.compile(optimizer=optimizer, loss="mse", metrics=["accuracy"])


# print(training_data_generator[0][0].shape)

training = model.fit(
    lookback_train_mm,
    lookforward_train,
    epochs=500,
    batch_size=16,
    verbose=2,
    validation_data=(lookback_validate_mm, lookforward_validate),
)

model.save("models/wind_prediction/env_sci_recursive_model_1.keras")

plt.plot(training.history["loss"])
plt.plot(training.history["val_loss"])
plt.title("train vs validation loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "validation"], loc="upper right", ncol=2)

plt.show()
