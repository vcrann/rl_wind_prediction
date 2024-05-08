import torch
from torch import nn


class WindLSTM(nn.Module):
    def __init__(
        self, batch_size, timesteps, feature_size, hidden_size, output_size, dropout
    ):
        super(WindLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.output_size = output_size
        self.feature_size = feature_size

        self.batch_norm = nn.BatchNorm1d(self.timesteps)
        self.lstm1 = nn.LSTMCell(feature_size, hidden_size[0])
        self.lstm2 = nn.LSTMCell(hidden_size[0], hidden_size[1])
        self.lstm3 = nn.LSTMCell(hidden_size[1], hidden_size[2])
        self.lstm4 = nn.LSTMCell(hidden_size[2], hidden_size[3])
        self.fc = nn.Linear(hidden_size[3], output_size)
        self.dropout = nn.Dropout(dropout)

    def init_hidden(self):
        h_t0 = torch.zeros(1, self.hidden_size[0], dtype=torch.float32)  # .to(device)
        c_t0 = torch.zeros(1, self.hidden_size[0], dtype=torch.float32)

        h_t1 = torch.zeros(1, self.hidden_size[1], dtype=torch.float32)
        c_t1 = torch.zeros(1, self.hidden_size[1], dtype=torch.float32)

        h_t2 = torch.zeros(1, self.hidden_size[2], dtype=torch.float32)
        c_t2 = torch.zeros(1, self.hidden_size[2], dtype=torch.float32)

        h_t3 = torch.zeros(1, self.hidden_size[3], dtype=torch.float32)
        c_t3 = torch.zeros(1, self.hidden_size[3], dtype=torch.float32)

        return [(h_t0, c_t0), (h_t1, c_t1), (h_t2, c_t2), (h_t3, c_t3)]

    def forward(self, input, hidden):
        input = self.batch_norm(input)
        outputs = []
        for b in range(self.batch_size):
            outputs_t = []

            for t in range(self.timesteps):
                input_t = input[b][t]
                input_t = input_t.view(1, self.feature_size)

                hidden[0] = self.lstm1(input_t, hidden[0])

                hidden[1] = self.lstm2(hidden[0][0], hidden[1])

                hidden[2] = self.dropout(
                    self.lstm3(hidden[1][0], hidden[2])
                )  # Dropout on 90 node layer

                hidden[3] = self.lstm4(hidden[2][0], hidden[3])

                output = hidden[3][0]

                outputs_t.append(output)

            outputs_t = torch.cat(outputs_t, dim=1)
            outputs_t = outputs_t.reshape(1, -1).squeeze()  # to "flatten"?
            outputs.append(outputs_t)

        outputs = torch.stack(outputs)
        # outputs = self.dropout(outputs)

        return outputs


model = WindLSTM(
    batch_size=1,
    timesteps=10,
    feature_size=5,
    hidden_size=(420, 180, 90, 48),
    output_size=1,
    dropout=0.96,
)

# # toy example training
# a = torch.arange(10039 * 4 * 68).reshape(1, 4, 68).type(torch.FloatTensor)
# batch_size = 32
# for epoch in range(10):
#     # a = torch.split a in batches of batch_size
#     hidden = model.init_hidden()
#     out = model(a, hidden)

num_epochs = 100
learning_rate = 0.00003

critereon = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


hidden = model.init_hidden()

for epoch in range(num_epochs):
    outputs = model.forward(input, hidden)
    optimizer.zero_grad()

    loss = critereon(outputs, target)

    loss.backward()

    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
