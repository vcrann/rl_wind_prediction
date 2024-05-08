import torch
from torch import nn
from torch.autograd import Variable

# class WindLSTM(nn.Module):
#     def __init__(
#         self, batch_size, timesteps, feature_size, hidden_size, output_size, dropout
#     ):
#         super(WindLSTM, self).__init__()
#         self.hidden_size = hidden_size
#         self.timesteps = timesteps
#         self.batch_size = batch_size
#         self.output_size = output_size
#         self.feature_size = feature_size

#         self.batch_norm = nn.BatchNorm1d(self.timesteps)
#         self.lstm1 = nn.LSTMCell(feature_size, hidden_size[0])
#         self.lstm2 = nn.LSTMCell(hidden_size[0], hidden_size[1])
#         self.lstm3 = nn.LSTMCell(hidden_size[1], hidden_size[2])
#         self.lstm4 = nn.LSTMCell(hidden_size[2], hidden_size[3])
#         self.fc1 = nn.Linear(hidden_size[3], hidden_size[3])
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size[3], output_size)
#         self.dropout = nn.Dropout(dropout)

#     def init_hidden(self):
#         h_t0 = torch.zeros(1, self.hidden_size[0], dtype=torch.float32)  # .to(device)
#         c_t0 = torch.zeros(1, self.hidden_size[0], dtype=torch.float32)

#         h_t1 = torch.zeros(1, self.hidden_size[1], dtype=torch.float32)
#         c_t1 = torch.zeros(1, self.hidden_size[1], dtype=torch.float32)

#         h_t2 = torch.zeros(1, self.hidden_size[2], dtype=torch.float32)
#         c_t2 = torch.zeros(1, self.hidden_size[2], dtype=torch.float32)

#         h_t3 = torch.zeros(1, self.hidden_size[3], dtype=torch.float32)
#         c_t3 = torch.zeros(1, self.hidden_size[3], dtype=torch.float32)

#         return [(h_t0, c_t0), (h_t1, c_t1), (h_t2, c_t2), (h_t3, c_t3)]

#     def forward(self, input, hidden):
#         # input = self.batch_norm(input)
#         outputs = []
#         for b in range(self.batch_size):
#             outputs_t = []

#             for t in range(self.timesteps):
#                 input_t = input[b][t]
#                 input_t = input_t.view(1, self.feature_size)

#                 hidden[0] = self.lstm1(input_t, hidden[0])

#                 hidden[1] = self.lstm2(hidden[0][0], hidden[1])

#                 # hidden[2] = self.dropout(
#                 #     self.lstm3(hidden[1][0], hidden[2])
#                 # )  # Dropout on 90 node layer

#                 hidden[2] = self.lstm3(hidden[1][0], hidden[2])

#                 hidden[3] = self.lstm4(hidden[2][0], hidden[3])

#                 output = hidden[3][0]

#                 outputs_t.append(output)

#             outputs_t = torch.cat(outputs_t, dim=1)
#             outputs_t = outputs_t.reshape(1, -1).squeeze()  # to "flatten"?
#             outputs.append(outputs_t)

#         outputs = torch.stack(outputs)
#         # outputs = self.dropout(outputs)

#         return outputs


# FOR MODEL_1
# class WindLSTM(nn.Module):
#     def __init__(self, timesteps, feature_size, hidden_size, output_size, dropout):
#         super(WindLSTM, self).__init__()
#         self.hidden_size = hidden_size
#         self.timesteps = timesteps
#         self.output_size = output_size
#         self.feature_size = feature_size

#         self.lstm1 = nn.LSTMCell(feature_size, hidden_size[0])
#         self.lstm2 = nn.LSTMCell(hidden_size[0], hidden_size[1])
#         self.lstm3 = nn.LSTMCell(hidden_size[1], hidden_size[2])
#         self.lstm4 = nn.LSTMCell(hidden_size[2], hidden_size[3])
#         self.relu = nn.ReLU()
#         self.fc = nn.Linear(hidden_size[3], output_size)
#         # self.dropout = nn.Dropout(dropout) #TODO add later

#     def init_hidden(self):
#         h_t0 = torch.zeros(1, self.hidden_size[0], dtype=torch.float32)  # .to(device)
#         c_t0 = torch.zeros(1, self.hidden_size[0], dtype=torch.float32)

#         h_t1 = torch.zeros(1, self.hidden_size[1], dtype=torch.float32)
#         c_t1 = torch.zeros(1, self.hidden_size[1], dtype=torch.float32)

#         h_t2 = torch.zeros(1, self.hidden_size[2], dtype=torch.float32)
#         c_t2 = torch.zeros(1, self.hidden_size[2], dtype=torch.float32)

#         h_t3 = torch.zeros(1, self.hidden_size[3], dtype=torch.float32)
#         c_t3 = torch.zeros(1, self.hidden_size[3], dtype=torch.float32)

#         return [(h_t0, c_t0), (h_t1, c_t1), (h_t2, c_t2), (h_t3, c_t3)]

#     def forward(self, input):
#         hidden = self.init_hidden()
#         # input = torch.split(input, 1, dim=0)

#         outputs = []
#         for input_t in input:
#             h_t, c_t = self.lstm1(input_t, hidden[0])
#             h_t2, c_t2 = self.lstm2(h_t, hidden[1])
#             h_t3, c_t3 = self.lstm3(h_t2, hidden[2])
#             h_t4, c_t4 = self.lstm4(h_t3, hidden[3])
#             pre_output = self.relu(h_t4)
#             output = self.fc(pre_output)
#             outputs += [output]
#         outputs = torch.cat(outputs, dim=0)
#         return outputs


class WindLSTM(nn.Module):
    def __init__(
        self, timesteps, feature_size, output_size, num_layers=1, hidden_size=420
    ):
        super(WindLSTM, self).__init__()
        self.timesteps = timesteps
        self.output_size = output_size
        self.feature_size = feature_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )  # lstm
        self.fc_1 = nn.Linear(420, 128)  # fully connected 1
        self.fc = nn.Linear(128, output_size)  # fully connected last layer

        self.relu = nn.ReLU()

    def forward(self, input):
        h_0 = Variable(
            torch.zeros(self.num_layers, input.size(0), self.hidden_size)
        )  # hidden state
        c_0 = Variable(
            torch.zeros(self.num_layers, input.size(0), self.hidden_size)
        )  # internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(
            input, (h_0, c_0)
        )  # lstm with input, hidden, and internal state
        # hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        hn = hn[-1, :, :].view(-1, self.hidden_size)  # for multi layer
        out = self.relu(hn)
        out = self.fc_1(out)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc(out)  # Final Output
        return out
