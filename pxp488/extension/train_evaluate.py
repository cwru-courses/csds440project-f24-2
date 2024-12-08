import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import update_ema_variables, sigmoid_rampup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# to mark the unlabelled data
NO_LABEL = -1

# This method is mix match implimentation
def mixmatch(x_labeled, y_labeled, x_unlabeled, model, T=0.5, K=2, alpha=0.75):
    """
    MixMatch implementation based on the provided pseudocode.
    """
    device = x_labeled.device

    # Apply data augmentation to labeled and unlabeled examples
    x_labeled_aug = x_labeled 
    # unlabeled data(K augmentations)
    x_unlabeled_aug = [x_unlabeled] * K  

    # Model predictions for augmented unlabeled examples
    with torch.no_grad():
        q_augment = [torch.softmax(model(u), dim=1) for u in x_unlabeled_aug]

    # Average the predictions across augmentations
    q_mean = torch.mean(torch.stack(q_augment), dim=0)

    # Apply sharpening to the predictions
    q_sharpen = sharpen(q_mean, T)

    # One-hot encode y_labeled
    num_classes = q_sharpen.size(1)
    y_labeled_onehot = torch.zeros(len(y_labeled), num_classes, device=device)
    y_labeled_onehot.scatter_(1, y_labeled.view(-1, 1), 1)

    # Combine labeled and unlabeled examples
    W = torch.cat([x_labeled, x_unlabeled], dim=0)
    W_labels = torch.cat([y_labeled_onehot, q_sharpen], dim=0)

    # Shuffle the combined data
    indices = torch.randperm(W.size(0))
    W, W_labels = W[indices], W_labels[indices]

    # Apply MixUp regularization
    x_labeled_prime, y_labeled_prime = mixup(x_labeled, W[:len(x_labeled)], W_labels[:len(x_labeled)], alpha)
    x_unlabeled_prime, _ = mixup(x_unlabeled, W[len(x_labeled):], W_labels[len(x_labeled):], alpha)

    return x_labeled_prime, y_labeled_prime, x_unlabeled_prime

# Apply temperature sharpening to predictions.
def sharpen(p, T):

    p = p ** (1 / T)
    return p / p.sum(dim=1, keepdim=True)

# MixUp regularization for inputs and labels.
def mixup(x, x_shuffled, y_shuffled, alpha):
  
    lam = np.random.beta(alpha, alpha)
    x_mixup = lam * x + (1 - lam) * x_shuffled
    y_mixup = lam * y_shuffled + (1 - lam) * y_shuffled
    return x_mixup, y_mixup

# This method is to train the semi-supervised learning (mean teacher - self learning) with mixmatch method
def train_ssl_mixmatch(train_loader, model, ema_model, optimizer, epoch, consistency_weight, rampup_length, global_step, T, K, alpha):
    model.train()
    ema_model.train()
    model.to(device)
    ema_model.to(device)

    # this is for cross entropy loss
    class_criterion = nn.CrossEntropyLoss(ignore_index=NO_LABEL).to(device)
    # this is for mean square error loss
    consistency_criterion = nn.MSELoss().to(device)

    total_loss = 0
    batch_count = 0  # Counter to keep track of the number of batches

    for (x_labeled, y_labeled), (x_unlabeled, _) in train_loader:
        x_labeled, y_labeled = x_labeled.to(device), y_labeled.to(device)
        x_unlabeled = x_unlabeled.to(device)

        # Apply MixMatch method
        x_labeled_prime, y_labeled_prime, x_unlabeled_prime = mixmatch(x_labeled, y_labeled, x_unlabeled, model, T, K, alpha)

        # Forward pass for labeled data
        outputs_labeled = model(x_labeled_prime)
        class_loss = class_criterion(outputs_labeled, y_labeled_prime)

        # Forward pass for unlabeled data
        outputs_unlabeled = model(x_unlabeled_prime)
        with torch.no_grad():
            ema_outputs = ema_model(x_unlabeled_prime)

        # Consistency loss
        consistency_loss = consistency_criterion(F.softmax(outputs_unlabeled, dim=1), F.softmax(ema_outputs, dim=1))

        # Ramp-up consistency weight
        consistency_weight_ramped = consistency_weight * sigmoid_rampup(epoch, rampup_length)

        # Total loss
        loss = class_loss + consistency_weight_ramped * consistency_loss
        total_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update EMA model
        update_ema_variables(model, ema_model, 0.99, global_step)
        global_step += 1
        # Increment the batch counter
        batch_count += 1 

    # Use batch_count instead of len(train_loader)
    print(f"Epoch {epoch}: Training Loss = {total_loss / batch_count:.4f}")
    return global_step

# Evaluation loop
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
