import tensorflow as tf
import keras
from keras import layers
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

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

model = keras.saving.load_model(
    "models/wind_prediction/env_sci_recursive_model_1.keras"
)
predicted_wind = model.predict(lookback_validate_mm)

true_wind = lookforward_validate


plt.figure(figsize=(10, 6))  # plotting
# plt.axvline(x=1800, c="r", linestyle="--")  # size of the training set

plt.plot(true_wind, label="Anemometer Windspeed [m/s]")  # actual plot
plt.plot(predicted_wind, label="Predicted Windspeed [m/s]")  # predicted plot
plt.title("Wind Speed Prediction")
plt.legend()


plt.show()
