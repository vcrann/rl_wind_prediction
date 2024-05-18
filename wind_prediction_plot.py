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

raw_data = raw_data[:10000]

lookback_window = 20
prediction_window = 10

mm = MinMaxScaler()
normalised_data = mm.fit_transform(raw_data)

model = keras.saving.load_model("models/wind_prediction/env_sci_model_1.keras")

plotting_indexes = [5000]

data_generator = DataGenerator(
    plotting_indexes,
    normalised_data,
    lookback_window,
    prediction_window,
    shuffle=True,
    batch_size=16,
)

lookback = np.array([data_generator[0][0][0]])
predicted_wind = model.predict(lookback)
predicted_wind = mm.inverse_transform(predicted_wind[0])
true_wind = mm.inverse_transform(data_generator[0][1][0])

predicted_wind = np.concatenate((mm.inverse_transform(lookback[0]), predicted_wind))
true_wind = np.concatenate((mm.inverse_transform(lookback[0]), true_wind))


plt.figure(figsize=(10, 6))  # plotting
plt.axvline(x=19, c="r", linestyle="--")  # size of the training set

plt.plot(true_wind, label="Anemometer Windspeed [m/s]")  # actual plot
plt.plot(predicted_wind, label="Predicted Windspeed [m/s]")  # predicted plot
plt.title("Wind Speed Estimation")

plt.legend()
plt.show()
