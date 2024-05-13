import tensorflow as tf
import keras
from keras import layers
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/training_data.csv")

uas_state_data = df.iloc[:, 3:]  # X
wind_data = df.iloc[:, 1:3]  # Y

ss = StandardScaler()
mm = MinMaxScaler()

state_ss = ss.fit_transform(uas_state_data)
wind_data_mm = mm.fit_transform(wind_data)

state_ss = np.expand_dims(state_ss, 1)

model = keras.saving.load_model("models/keras/model_keras_3.keras")
predicted_wind = model.predict(state_ss)

predicted_wind = mm.inverse_transform(predicted_wind)

true_wind = mm.inverse_transform(wind_data_mm)


plt.figure(figsize=(10, 6))  # plotting
plt.axvline(x=1800, c="r", linestyle="--")  # size of the training set

plt.plot(true_wind[:, 0], label="Anemometer Windspeed [m/s]")  # actual plot
plt.plot(predicted_wind[:, 0], label="Predicted Windspeed [m/s]")  # predicted plot
plt.title("Wind Speed Estimation")

# plt.plot(dataY_plot, label="Actual Data")  # actual plot
# plt.plot(data_predict, label="Predicted Data")  # predicted plot
plt.title("Wind Speed Prediction")
plt.legend()

plt.figure(figsize=(10, 6))  # plotting
plt.axvline(x=1800, c="r", linestyle="--")  # size of the training set
plt.plot(true_wind[:, 1], label="Anemometer Wind Direction [°]")  # actual plot
plt.plot(predicted_wind[:, 1], label="Predicted Wind Direction [°]")  # predicted plot
plt.title("Wind Direction Estimation")
plt.legend()
plt.show()
