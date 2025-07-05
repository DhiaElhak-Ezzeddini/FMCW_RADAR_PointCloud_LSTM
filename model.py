from read_gestures import load_data 
import numpy as np 
import matplotlib.pyplot as plt
from torch.utils.data import random_split

import torch 
from torch.utils.data import TensorDataset , DataLoader
import torch.nn as nn
import torch.optim as optim

from read_gestures_test import load_data_exp
from itertools import chain

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


class RadarLstm(nn.Module):
    def __init__(self , input_size , hidden_size , num_layers , num_classes , dropout:float=0.2): 
        super().__init__()
        
        self.lstm = nn.LSTM(input_size , hidden_size , num_layers , batch_first=True , bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size*2 , num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        h_lstm,_ = self.lstm(x)
        output = h_lstm[:,1,:]
        output = self.dropout(output)
        output = self.fc(output)
        output = self.logsoftmax(output)
        
        return output
    
    
data , labels = load_data()

data   = np.array(data)
labels = np.array(labels)

data_tensor   = torch.tensor(data,dtype=torch.float32)
labels_tensor = torch.tensor(labels,dtype=torch.long)

dataset = TensorDataset(data_tensor,labels_tensor) ## dataset ready to do the job :)

## Splitting the dataset into train and test data 90% / 10% 
train_size = int(0.8*len(dataset))
test_size  = len(dataset) - train_size

train_data , test_data = random_split(dataset,[train_size,test_size])

train_loader = DataLoader(train_data , batch_size=32 , shuffle=True)
test_loader = DataLoader(test_data , batch_size=32 , shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device : {device}")

from collections import Counter
print(Counter(labels_tensor.tolist()))

print(f'Full dataset size: {len(dataset)}')
print(f'Train dataset size: {len(train_data)}')
print(f'Test dataset size: {len(test_data)}')
print("")
sample, label = dataset[0]
print(f'Input sample shape: {sample.size()}')  # (80, 240)
print(f'Label shape: {label.size()}')          # ()

print("")
for batch_idx, (inputs, targets) in enumerate(train_loader):
    print(f"Batch {batch_idx}: Inputs shape {targets.shape}")
    print(inputs[15])
    print(targets[1])
    if batch_idx==1 : 
        break
    
### Defining the model 
model = RadarLstm(input_size=160 , hidden_size=128 , num_layers=2 , num_classes=12 , dropout=0.2)
model = model.to(device)
print(model)

criterion =nn.NLLLoss() ## LogSoftmax --> NLLoss
optimizer = optim.Adam(model.parameters() , lr=0.001)


num_epochs = 50


for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    batch_count = 0
    for batch_data , batch_labels in train_loader :
        
        optimizer.zero_grad()
        batch_data   = batch_data.to(device)
        batch_labels = batch_labels.to(device)
    
        outputs = model(batch_data)
        loss    = criterion(outputs , batch_labels)
        
        loss.backward() 
        optimizer.step()
        total_loss += loss.item()
        batch_count += 1
    avg_loss = total_loss / batch_count
    print(f"Epoch [{epoch+1}/{num_epochs}] -- {batch_count} batches processed -- Average Loss: {avg_loss:.4f}")


## Evaluation 
model.eval()
all_preds    = []
all_targets = []

with torch.no_grad():
    for batch_udx , (inputs , targets) in enumerate(test_loader) : 
        inputs = inputs.to(device)
        targets = targets.to(device) 
        
        outs = model(inputs)
        _,predicted = torch.max(outs,1)
        all_preds.append(predicted.cpu())
        all_targets.append(targets.cpu())
        
all_preds = torch.cat(all_preds)    
all_targets = torch.cat(all_targets)

y_true = np.array(all_targets)
y_pred = np.array(all_preds)
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision (macro):", precision_score(y_true, y_pred, average='macro'))
print("Recall (macro):", recall_score(y_true, y_pred, average='macro'))
print("F1 Score (macro):", f1_score(y_true, y_pred, average='macro'))

print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred))

cm = confusion_matrix(all_targets,all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)    
plt.show()

## Evaluation with the new measured hand gesture samples ==> "hand closer – taking a hand closer to the radar"

data , labels = load_data_exp()

data   = np.array(data)
labels = np.array(labels)

data_tensor   = torch.tensor(data,dtype=torch.float32)
labels_tensor = torch.tensor(labels,dtype=torch.long)

dataset = TensorDataset(data_tensor,labels_tensor)

exp_loader = DataLoader(dataset , batch_size=3 , shuffle=True)


model.eval()
all_preds    = []
all_targets = []

with torch.no_grad():
    for batch_udx , (inputs , targets) in enumerate(chain(test_loader,  exp_loader)) : 
        inputs = inputs.to(device)
        targets = targets.to(device) 

        
        outs = model(inputs)
        _,predicted = torch.max(outs,1)
        all_preds.append(predicted.cpu())
        all_targets.append(targets.cpu())
        
all_preds = torch.cat(all_preds)    
all_targets = torch.cat(all_targets)

y_true = np.array(all_targets)
y_pred = np.array(all_preds)
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision (macro):", precision_score(y_true, y_pred, average='macro'))
print("Recall (macro):", recall_score(y_true, y_pred, average='macro'))
print("F1 Score (macro):", f1_score(y_true, y_pred, average='macro'))

print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred))

cm = confusion_matrix(all_targets,all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)    
plt.show()

