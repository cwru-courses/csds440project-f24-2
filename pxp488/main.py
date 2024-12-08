import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from train_evaluate import train_ssl, evaluate
from architecture import CNN

def main():
    # These are the Hyperparameters
    batch_size = 128
    num_epochs = 20
    learning_rate = 1e-3
    consistency_weight = 1.0
    rampup_length = 5
    global_step = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # To prepare Dataset 
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
    os.makedirs(dataset_path, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    #MNIST dataset
    dataset = datasets.MNIST(root=dataset_path, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=dataset_path, train=False, transform=transform, download=True)
    model = CNN(num_classes=10).to(device)

    # Split the given data into labeled data and unlabeled dataset
    labeled_size = int(len(dataset) * 0.1)
    unlabeled_size = len(dataset) - labeled_size
    labeled_data, unlabeled_data = random_split(dataset, [labeled_size, unlabeled_size])

    labeled_loader = DataLoader(labeled_data, batch_size=batch_size, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # EMA Model which is teacher model
    ema_model = CNN(num_classes=10).to(device)

    # model Optimizer definition
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # This loop is for Training and validation 
    for epoch in range(1, num_epochs + 1):
        global_step = train_ssl(labeled_loader, model, ema_model, optimizer, epoch, consistency_weight, rampup_length, global_step)
        val_accuracy = evaluate(model, test_loader)
        print(f"Epoch {epoch:02d}: Validation Accuracy: {val_accuracy:.2f}%")

# main function calling
if __name__ == "__main__":
    main()
