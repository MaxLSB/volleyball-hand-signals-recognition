import torch

# Each npy file contains 33*4 + 21*3 + 21*3 = 258 values

class DetectionModel(torch.nn.Module):
    def __init__(self, num_actions, input_size=258):
        self.input_size = input_size
        self.num_actions = num_actions
        super().__init__()
        # batch_first = True => batch_size, sequence_length, input_size
        self.lstm1 = torch.nn.LSTM(self.input_size, hidden_size=64, num_layers=1, batch_first=True)
        self.lstm2 = torch.nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        self.lstm3 = torch.nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1 = torch.nn.Linear(64, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, self.num_actions)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.relu(x)
        x, _ = self.lstm2(x)
        x = self.relu(x)
        x, _ = self.lstm3(x)
        # We only take the last element of the sequence
        x = self.relu(x[:, -1, :])  # Shape is now (batch_size, input_size)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x