import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils import transform, test_transform, update_ema_variables, sigmoid_rampup
from architecture import CNN
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# to mark the unlabelled data
NO_LABEL = -1  

# This method is to train the semi-supervised learning (mean teacher - self learning)
def train_ssl(train_loader, model, ema_model, optimizer, epoch, consistency_weight, rampup_length, global_step):
    model.train()
    ema_model.train()
    model.to(device)
    ema_model.to(device)

    # this is for cross entropy loss
    class_criterion = nn.CrossEntropyLoss(ignore_index=NO_LABEL).to(device)
    # this is for mean square error loss
    consistency_criterion = nn.MSELoss().to(device)

    total_loss = 0

    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        with torch.no_grad():
            ema_outputs = ema_model(inputs)

        # Supervised classification loss
        labeled_mask = targets != NO_LABEL
        if labeled_mask.sum() > 0:
            class_loss = class_criterion(outputs[labeled_mask], targets[labeled_mask])
        else:
            class_loss = torch.tensor(0.0).to(device)

        # Consistency loss
        consistency_loss = consistency_criterion(F.softmax(outputs, dim=1), F.softmax(ema_outputs, dim=1))

        # Ramp-up consistency weight
        consistency_weight_ramped = consistency_weight * sigmoid_rampup(epoch, rampup_length)

        # this is the Total loss
        loss = class_loss + consistency_weight_ramped * consistency_loss
        total_loss += loss.item()

        # this is the Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # this is to Update EMA model
        update_ema_variables(model, ema_model, 0.99, global_step)
        global_step += 1

    print(f"Epoch {epoch}: Training Loss = {total_loss / len(train_loader):.4f}")
    return global_step

# this method is to evaluate the model validation accuracy
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = 100 * correct / total
    return accuracy
