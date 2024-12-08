import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from train_evaluate import train_ssl_mixmatch, evaluate
from architecture import CNN

def main():
    # These are the Hyperparameters
    batch_size = 128
    num_epochs = 20
    learning_rate = 1e-3
    consistency_weight = 1.0
    rampup_length = 5
    global_step = 0
    T = 0.5
    K = 2
    alpha = 0.75
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # To prepare Dataset 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    #MNIST dataset
    dataset = datasets.MNIST(root="data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="data", train=False, transform=transform, download=True)

    # Split the given data into labeled data and unlabeled dataset
    labeled_size = int(len(dataset) * 0.1)
    unlabeled_size = len(dataset) - labeled_size
    labeled_data, unlabeled_data = random_split(dataset, [labeled_size, unlabeled_size])

    labeled_loader = DataLoader(labeled_data, batch_size=batch_size, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Models and optimizer 
    model = CNN(num_classes=10).to(device)
    ema_model = CNN(num_classes=10).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Training and evaluation loop
    for epoch in range(1, num_epochs + 1):
        global_step = train_ssl_mixmatch(
            zip(labeled_loader, unlabeled_loader),
            model,
            ema_model,
            optimizer,
            epoch,
            consistency_weight,
            rampup_length,
            global_step,
            T,
            K,
            alpha
        )

        val_accuracy = evaluate(model, test_loader)
        print(f"Epoch {epoch:02d}: Validation Accuracy: {val_accuracy:.2f}%")

if __name__ == "__main__":
    main()
