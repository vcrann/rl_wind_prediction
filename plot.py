import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from model import WindLSTM
import matplotlib.pyplot as plt

df = pd.read_csv("data/training_data.csv")

mm = MinMaxScaler()
ss = StandardScaler()

uas_state_data = df.iloc[:, 3:]  # X
wind_data = df.iloc[:, 1:3]  # Y

df_state_ss = ss.fit_transform(uas_state_data)
df_wind_mm = mm.fit_transform(wind_data)

# df_state_ss = StandardScaler().transform(df.iloc[:, 3:])
# df_wind_mm = MinMaxScaler().transform(df.iloc[:, 1:3])

df_state_ss = Variable(torch.Tensor(df_state_ss))
df_wind_mm = Variable(torch.Tensor(df_wind_mm))

df_state_ss = torch.reshape(
    df_state_ss, (df_state_ss.shape[0], 1, df_state_ss.shape[1])
)

# For model 1
# model = WindLSTM(
#     timesteps=10,
#     feature_size=6,
#     hidden_size=(420, 180, 90, 48),
#     output_size=2,
#     dropout=0.96,
# )

# For model 2/3
model = WindLSTM(
    timesteps=10,
    feature_size=6,
    output_size=2,
    num_layers=2,
)
model.load_state_dict(torch.load("models/model_3.pt"))
model.eval()

train_predict = model(df_state_ss)  # forward pass
data_predict = train_predict.data.numpy()  # numpy conversion
wind_data_plot = df_wind_mm.data.numpy()

data_predict = mm.inverse_transform(data_predict)  # reverse transformation
print(data_predict[:, 0])
dataY_plot = mm.inverse_transform(wind_data_plot)
plt.figure(figsize=(10, 6))  # plotting
plt.axvline(x=1800, c="r", linestyle="--")  # size of the training set

plt.plot(dataY_plot[:, 0], label="Anemometer Windspeed [m/s]")  # actual plot
plt.plot(data_predict[:, 0], label="Predicted Windspeed [m/s]")  # predicted plot
plt.title("Wind Speed Estimation")

# plt.plot(dataY_plot, label="Actual Data")  # actual plot
# plt.plot(data_predict, label="Predicted Data")  # predicted plot
plt.title("Wind Speed Prediction")
plt.legend()

plt.figure(figsize=(10, 6))  # plotting
plt.axvline(x=1800, c="r", linestyle="--")  # size of the training set
plt.plot(dataY_plot[:, 1], label="Anemometer Wind Direction [°]")  # actual plot
plt.plot(data_predict[:, 1], label="Predicted Wind Direction [°]")  # predicted plot
plt.title("Wind Direction Estimation")
plt.legend()
plt.show()
