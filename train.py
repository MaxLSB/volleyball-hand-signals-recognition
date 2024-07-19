import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import utils.preProcess as preProcess
from models.model import DetectionModel

DATA_PATH = os.path.join('data') # Path for exported data, numpy arrays
actions = np.array(['pointL', 'pointR']) # Actions that we try to detect
no_sequences = 30 # Thirty videos worth of data
sequence_length = 30 # Videos are going to be 30 frames in length
start_folder = 1 # Folder start
num_epochs=30
batch_size=8

sequences, labels = preProcess.pre_processing(actions, DATA_PATH, sequence_length)
X = np.array(sequences)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

model = DetectionModel(num_actions=len(actions))

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())



for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Saving the model
torch.save(model.state_dict(), 'trained_model/refereeSignalsModel.pth')
print('Model saved as refereeSignalsModel.pth')