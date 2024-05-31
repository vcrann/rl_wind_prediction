import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm
from pmdarima.model_selection import train_test_split
from ssa import SSA

raw_data = np.load("data/earth_sciences_anemometer/worst_day.npy")  # shape: (x, 3)

y_wind = raw_data[:, 1]
y_wind = y_wind[: len(y_wind) - len(y_wind) % 20]

average_n = 20

avg_y_wind = np.average(y_wind.reshape(-1, average_n), axis=1)

avg_y_wind_10m = avg_y_wind[:600]

avg_y_wind_SSA = SSA(avg_y_wind_10m, 10)
# avg_y_wind_SSA.components_to_df().plot()
# avg_y_wind_SSA.orig_TS.plot(alpha=0.4)
# plt.show()

avg_y_wind_DN = avg_y_wind_SSA.components_to_df()["F0"].to_numpy()


train, test = train_test_split(avg_y_wind_DN, test_size=10)

# Fit the model

# model = pm.auto_arima(
#     train,
#     start_p=1,
#     start_q=1,
#     test="adf",
#     max_p=10,
#     max_q=10,
#     m=1,
#     d=1,
#     seasonal=False,
#     start_P=0,
#     D=None,
#     trace=True,
#     error_action="ignore",
#     suppress_warnings=True,
#     stepwise=False,
# )


model = pm.arima.ARIMA(order=(3, 1, 1))
model.fit(train)

prediction = model.predict(test.shape[0])  # predict N steps into the future

x = np.arange(avg_y_wind_10m.shape[0])
plt.figure(figsize=(10, 6))  # plotting
plt.plot(avg_y_wind_10m, label="Avg Anemometer Windspeed y [m/s]")  # actual plot
plt.plot(avg_y_wind_DN, label="F0 [m/s]")  # actual plot
plt.plot(x[len(train) :], prediction, label="Prediction")  # predicted plot
plt.title("Wind Speed Prediction (y)")
plt.legend()
plt.show()
