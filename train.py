import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from model import WindLSTM

# Prepare the data
df = pd.read_csv("data/training_data.csv")

uas_state_data = df.iloc[:, 3:]  # X
wind_data = df.iloc[:, 1:3]  # Y

state_ss = StandardScaler().fit_transform(uas_state_data)
wind_data_mm = MinMaxScaler().fit_transform(wind_data)

state_train = state_ss[:1800, :]
state_test = state_ss[1800:, :]

wind_train = wind_data_mm[:1800, :]
wind_test = wind_data_mm[1800:, :]

state_train_tensors = Variable(torch.Tensor(state_train))
state_test_tensors = Variable(torch.Tensor(state_test))

wind_train_tensors = Variable(torch.Tensor(wind_train))
wind_test_tensors = Variable(torch.Tensor(wind_test))

# Reshape for LSTM

state_train_tensors = torch.reshape(
    state_train_tensors, (state_train_tensors.shape[0], 1, state_train_tensors.shape[1])
)

state_test_tensors = torch.reshape(
    state_test_tensors, (state_test_tensors.shape[0], 1, state_test_tensors.shape[1])
)


# model = WindLSTM(
#     timesteps=10,
#     feature_size=6,
#     hidden_size=(420, 180, 90, 48),
#     output_size=2,
#     dropout=0.96,
# )

model = WindLSTM(
    timesteps=10,
    feature_size=6,
    output_size=2,
    num_layers=2,
)


# # toy example training
# a = torch.arange(10039 * 4 * 68).reshape(1, 4, 68).type(torch.FloatTensor)
# batch_size = 32
# for epoch in range(10):
#     # a = torch.split a in batches of batch_size
#     hidden = model.init_hidden()
#     out = model(a, hidden)

num_epochs = 1000
# learning_rate = 0.00003
learning_rate = 0.0001

critereon = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    outputs = model.forward(state_train_tensors)
    optimizer.zero_grad()

    loss = critereon(outputs, wind_train_tensors)

    loss.backward()

    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

# Save the model
torch.save(model.state_dict(), "models/model_3.pt")
