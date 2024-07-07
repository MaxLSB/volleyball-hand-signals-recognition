import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
import src.preProcess as preProcess
import src.model as model

DATA_PATH = os.path.join('data') # Path for exported data, numpy arrays
actions = np.array(['pointLeft', 'pointRight']) # Actions that we try to detect
no_sequences = 30 # Thirty videos worth of data
sequence_length = 30 # Videos are going to be 30 frames in length
start_folder = 30 # Folder start

sequences, labels = preProcess.pre_processing(actions, DATA_PATH, sequence_length)
X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = model.building_model(actions, X_train, y_train, tb_callback)
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.save('refereeSignalsModel.h5')

