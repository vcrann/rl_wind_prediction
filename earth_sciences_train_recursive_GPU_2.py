import numpy as np
import keras
from keras import layers
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Paths for colab
lookback = np.load(
    "/content/drive/MyDrive/rl_wind_prediction_data/earth_sciences_anemometer/training_data/worst_day_X_20_1.npy"
)  # Lookback
lookforward = np.load(
    "/content/drive/MyDrive/rl_wind_prediction_data/earth_sciences_anemometer/training_data/worst_day_Y_20_1.npy"
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

lookforward_train = np.squeeze(lookforward_train)

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
model.add(layers.LSTM(64, activation="tanh", dropout=0.96))
model.add(layers.Dense(data_dim))
# model.compile(optimizer="adam", loss="mse", metrics=["accuracy"], learning_rate=4e-5)
optimizer = keras.optimizers.RMSprop(learning_rate=4e-5)
model.compile(optimizer=optimizer, loss="mse", metrics=["accuracy"])


training = model.fit(
    lookback_train_mm,
    lookforward_train,
    epochs=500,
    batch_size=16,
    verbose=2,
    shuffle=True,
    validation_data=(lookback_validate_mm, lookforward_validate),
)

model.save("models/wind_prediction/env_sci_recursive_model_2.keras")

plt.plot(training.history["loss"])
plt.plot(training.history["val_loss"])
plt.title("train vs validation loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "validation"], loc="upper right", ncol=2)

plt.show()
