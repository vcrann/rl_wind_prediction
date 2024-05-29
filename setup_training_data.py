import math
import numpy as np

raw_data = np.load("data/earth_sciences_anemometer/worst_day.npy")  # shape: (x, 3)

lookback_window = 20
prediction_window = 1

# output_size = raw_data.shape[0] - lookback_window - prediction_window + 1

output_size = math.floor(raw_data.shape[0] / (lookback_window + prediction_window))

# X = np.empty((output_size, lookback_window, raw_data.shape[1]))
# Y = np.empty((output_size, prediction_window, raw_data.shape[1]))
X = np.empty((output_size, lookback_window, raw_data.shape[1]))
Y = np.empty((output_size, prediction_window, raw_data.shape[1]))

step_size = lookback_window + prediction_window

for i in range(output_size):
    X[i] = raw_data[
        i * step_size : i * step_size + lookback_window
    ]  # shape: (lookback_window, 3)
    Y[i] = raw_data[
        i * step_size
        + lookback_window : i * step_size
        + lookback_window
        + prediction_window
    ]  # shape: (prediction_window, 3)

# for i in range(
#     lookback_window, raw_data.shape[0] - prediction_window + 1):
#     X[i - lookback_window] = raw_data[
#         i - lookback_window : i
#     ]  # shape: (lookback_window, 3)
#     Y[i - lookback_window] = raw_data[
#         i : i + prediction_window
#     ]  # shape: (prediction_window, 3)

print(X.shape)
print(Y.shape)

np.save("data/earth_sciences_anemometer/training_data/worst_day_X_20_1.npy", X)
np.save("data/earth_sciences_anemometer/training_data/worst_day_Y_20_1.npy", Y)
