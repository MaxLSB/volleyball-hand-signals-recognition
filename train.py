import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import utils.preProcess as preProcess
from models.model import DetectionModel
from utils.actions import all_actions
from utils.config import get_paremeters

DATA_PATH = get_paremeters()['DATA_PATH']
actions = all_actions() # Actions that we try to detect
sequence_length = get_paremeters()['sequence_length'] # Videos are going to be 30 frames in length
num_epochs = get_paremeters()['num_epochs']
batch_size = get_paremeters()['batch_size']

print('Loading data...')
sequences, labels = preProcess.pre_processing(actions, DATA_PATH, sequence_length)

X = torch.tensor(sequences, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.long)
train_dataset = TensorDataset(X, y)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

model = DetectionModel(num_actions=len(actions))

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        
        # Returns the maximum value of each sample in the batch of outputs and the index of the maximum value (predicted class)
        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = correct_preds / total_preds
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

# Saving the model
torch.save(model.state_dict(), 'trained_model/refereeSignalsModel.pth')
print('Model saved as refereeSignalsModel.pth')