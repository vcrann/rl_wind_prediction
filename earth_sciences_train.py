import numpy as np
import random
import keras
from keras import layers
from data_generator import DataGenerator
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

raw_data = np.load("data/earth_sciences_anemometer/all_data.npy")
day_change_indexes = np.load(
    "data/earth_sciences_anemometer/end_indexes.npy"
)  # Last index of each day

# Making the dataset smaller for development
raw_data = raw_data[:100000]
day_change_indexes = day_change_indexes[
    : day_change_indexes.searchsorted(100000, "right") - 1
]

data_indexes = list(range(len(raw_data)))

lookback_window = 200
prediction_window = 20

day_start_index = 0
for day_end_index in day_change_indexes:
    # prediction_indexes = list(range(day_end_index - prediction_window + 1, day_end_index))
    del data_indexes[
        day_end_index - prediction_window + 1 : day_end_index
    ]  # remove indexes that cant be selected at end of day
    # lookback_indexes = list(
    #     range(day_start_index, day_start_index + lookback_window - 1)
    # )
    del data_indexes[
        day_start_index : day_start_index + lookback_window - 1
    ]  # remove indexes that cant be selected at start of day
    day_start_index = day_end_index + 1

training_indexes = data_indexes[: int(len(data_indexes) * 0.8)]
validation_indexes = data_indexes[int(len(data_indexes) * 0.8) :]

# TODO this is a bit of a sin because it causes data leakage, should scale training and validation data separately
mm = MinMaxScaler()
normalised_data = mm.fit_transform(raw_data)


random.shuffle(training_indexes)
training_data_generator = DataGenerator(
    training_indexes,
    normalised_data,
    lookback_window,
    prediction_window,
    16,
    use_multiprocessing=True,
    workers=8,
)

validation_data_generator = DataGenerator(
    validation_indexes,
    normalised_data,
    lookback_window,
    prediction_window,
    16,
    use_multiprocessing=True,
    workers=8,
)

data_dim = 3
model = keras.Sequential()
model.add(keras.Input(shape=(None, data_dim)))
# model.add(layers.LSTM(420, return_sequences=True, input_shape=(None, data_dim)))
model.add(layers.LSTM(420, return_sequences=True))
model.add(layers.LSTM(180, return_sequences=True))
model.add(
    layers.LSTM(90, return_sequences=True, dropout=0.96)
)  # Check this droput, seems high?
model.add(layers.LSTM(60, activation="relu"))  # changed from 48
model.add(layers.Dense(data_dim * prediction_window))
model.add(layers.Reshape((prediction_window, data_dim)))
# model.compile(optimizer="adam", loss="mse", metrics=["accuracy"], learning_rate=4e-5)
optimizer = keras.optimizers.RMSprop(learning_rate=4e-5)
model.compile(optimizer=optimizer, loss="mse", metrics=["accuracy"])


# print(training_data_generator[0][0].shape)
# TODO normalise data!
training = model.fit(
    training_data_generator,
    epochs=1500,
    batch_size=16,
    verbose=2,
    validation_data=validation_data_generator,
)

plt.plot(training.history["loss"])
plt.plot(training.history["val_loss"])
plt.title("train vs validation loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "validation"], loc="upper right", ncol=2)

plt.show()

model.save("models/wind_prediction/env_sci_model_1.keras")
