# -*- coding: utf-8 -*-
"""lstm_nucleus_smotenc.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZOFSklAHI8NKD5zxoeMy2z-3OCVLj5nd

# Nucleus-level LSTM Classifier (with artificially balanced data set)

## Import necessary packages
"""

import os
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, PrecisionRecallDisplay

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

"""## Set working directory and load datasets"""

from google.colab import drive
drive.mount('/content/drive')

os.chdir("/content/drive/My Drive/Colab Notebooks/")

data_features = np.load("data/nucleus_level_smotenc/nucleus_features_smotenc.npy", allow_pickle=True)
data_labels = np.load("data/nucleus_level_smotenc/nucleus_labels_smotenc.npy", allow_pickle=True)
data_features_unbalanced = np.load("data/nucleus_level/nucleus_features.npy", allow_pickle=True)
data_labels_unbalanced = np.load("data/nucleus_level/nucleus_labels.npy", allow_pickle=True)

"""## Split into training and testing sets"""

train_features, val_features, train_labels, val_labels = train_test_split(data_features, data_labels, test_size=0.2, random_state=1)
train_features_unbalanced, val_features_unbalanced, train_labels_unbalanced, val_labels_unbalanced = train_test_split(data_features_unbalanced, data_labels_unbalanced, test_size=0.2, random_state=1)

def to_tensor(data_list):
    return [torch.FloatTensor(data) for data in data_list]

train_features_input = pad_sequence(to_tensor(train_features), batch_first=True)
val_features_input = pad_sequence(to_tensor(val_features), batch_first=True)
train_labels_input = pad_sequence(to_tensor(train_labels), batch_first=True)
val_labels_input = pad_sequence(to_tensor(val_labels), batch_first=True)

train_features_unbalanced_input = pad_sequence(to_tensor(train_features_unbalanced), batch_first=True)
val_features_unbalanced_input = pad_sequence(to_tensor(val_features_unbalanced), batch_first=True)
train_labels_unbalanced_input = pad_sequence(to_tensor(train_labels_unbalanced), batch_first=True)
val_labels_unbalanced_input = pad_sequence(to_tensor(val_labels_unbalanced), batch_first=True)

print(train_features_input.shape)
print(train_labels_input.shape)
print(train_features_unbalanced_input.shape)
print(train_labels_unbalanced_input.shape)

"""## Run sets as PyTorch Datasets through PyTorch DataLoaders

### Sets to PyTorch Datasets
"""

class trainData(Dataset):
    
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    
    def __len__(self):
        return len(self.features)

train_data = trainData(torch.FloatTensor(train_features_input), torch.FloatTensor(train_labels_input))

"""## Set basic training parameters"""

EPOCHS = 50
BATCH_SIZE = 11
LEARNING_RATE = 1e-03

LENGTH = train_features_input.shape[0]
NUM_FEATURES = train_features_input.shape[2]
SEQUENCE = train_features_input.shape[1]

"""### Initialize DataLoaders"""

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

"""## Ready GPU if available"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

"""## Define neural network"""

class binaryClassifier(nn.Module):
    
    def __init__(self):
        super(binaryClassifier, self).__init__() # initialize parent class

        self.lstm_1 = nn.LSTM(input_size=NUM_FEATURES, hidden_size=128, num_layers=2, bidirectional=True, batch_first=True)

        self.dense_1 = nn.Linear(in_features=256, out_features=32)
        self.dense_2 = nn.Linear(in_features=32, out_features=1)

        self.relu = nn.ReLU()
        self.dropout_1 = nn.Dropout(0.4)
        self.dropout_2 = nn.Dropout(0.4)
                
    def forward(self, inputs):
        lstm_out, (h, c) = self.lstm_1(inputs)
        x, lengths = pad_packed_sequence(lstm_out, batch_first=True, total_length=3522)

        x = self.relu(x)
        x = self.dropout_1(x)
        x = self.dense_1(x)
        x = self.relu(x)
        x = self.dropout_2(x)
        x = self.dense_2(x)

        return x

"""## Initialize model and move to GPU if available"""

model = binaryClassifier()
model.to(device)

"""## Define loss function and optimizer"""

loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3])).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

"""## Define function to determine real sequence length and define metrics function"""

def get_seq_len(seq):
  with torch.no_grad():
    lens = []
    for e in seq:
      e_no_0 = len(e[torch.where(~torch.all(torch.isclose(e, torch.cuda.FloatTensor([0])), axis=1))])
      lens.append(e_no_0)
    return lens

def collect_metrics(preds, lengths, labels):
  act_sig = nn.Sigmoid()
  preds = np.concatenate([act_sig(preds[i, :lengths[i], :]).cpu().detach().numpy() for i in range(len(preds))]).flatten()
  preds_bin = np.where(preds > 0.5, 1, 0)
  labels = np.array(np.concatenate([labels[i, :lengths[i]].cpu().detach().numpy() for i in range(len(labels))]), dtype=int).flatten()
  return preds_bin, labels

"""## Train model"""

model_num = 2

model.train()
max_e_f1 = 0

for e in range(1, EPOCHS + 1):
    e_loss = e_f1 = 0
    e_preds_bin = e_labels = np.array([])
    
    for feature_batch, label_batch in train_loader:
        
        torch.autograd.set_detect_anomaly(True)
        feature_batch, label_batch = feature_batch.to(device), label_batch.to(device)
        optimizer.zero_grad()

        input_lengths = get_seq_len(feature_batch)

        input_features = pack_padded_sequence(feature_batch, input_lengths, batch_first=True, enforce_sorted=False)
        
        pred_labels = model(input_features)

        loss = loss_func(pred_labels, label_batch.unsqueeze(2))
        
        loss.backward()
        optimizer.step()
        
        e_loss += loss.item()

        preds_bin, labels = collect_metrics(pred_labels, input_lengths, label_batch)
        e_labels = np.concatenate([e_labels, labels])
        e_preds_bin = np.concatenate([e_preds_bin, preds_bin])

    print(classification_report(e_labels, e_preds_bin, digits=4))
    results = classification_report(e_labels, e_preds_bin, output_dict=True)

    e_f1 = results["1.0"]["f1-score"]

    print(f'Epoch {e+0:03}: | Loss: {e_loss/len(train_loader):.5f}')
    if e_f1 > max_e_f1:
      for file in glob(f"model_store/nucleus_level_smotenc/model-{model_num}*"):
        os.remove(file)
      
      torch.save(model, f"model_store/nucleus_level_smotenc/model-{model_num}_epoch-{e}_f1-{e_f1:.3f}.pt")
      max_e_f1 = e_f1
    else:
      pass

"""## Evaluate model"""

model_file = "model-2_epoch-41_f1-0.767.pt"
eval_model = torch.load(f"model_store/nucleus_level_smotenc/{model_file}").eval()

val_data = trainData(torch.FloatTensor(val_features_unbalanced_input), torch.FloatTensor(val_labels_unbalanced_input))
val_loader = DataLoader(dataset=val_data, batch_size=11, shuffle=True)

for feature_batch, label_batch in val_loader:
  feature_batch = feature_batch.to(device)
  input_lengths = get_seq_len(feature_batch)
  input_features = pack_padded_sequence(feature_batch, input_lengths, batch_first=True, enforce_sorted=False)
  
  
  y_pred = eval_model(input_features)
  y_pred_bin, y_true = collect_metrics(y_pred, input_lengths, label_batch)

label_batch.shape

report = classification_report(y_true, y_pred_bin, digits=4)
print(report)
with open(f"eval/nucleus_level_smotenc/{model_file}.txt", "w") as reportfile:
  reportfile.write(report)