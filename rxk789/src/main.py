import os
import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from model import MNISTVariationalAutoEncoder, CIFAR10VariationalAutoEncoder
from utils import validate, optimized_ELBO_labeled, optimized_ELBO_unlabeled, m2_loss_labeled, m2_loss_unlabeled
import torch.nn.functional as F


def train(model, labeled_loader, unlabeled_loader, optimizer, criterion_labeled, criterion_unlabeled, epoch, device):
    """
    this function trains the model using both the labeled and unlabeled data.
    """
    model.train()
    tot_loss = 0
    # using zip fuction to create tuples of labeled and unlabed data and iterate over it with bath size in (x_labeled, y_labeled), (x_unlabeled, _) 
    for (x_labeled, y_labeled), (x_unlabeled, _) in zip(labeled_loader, unlabeled_loader):
        x_labeled = x_labeled.to(device)
        y_labeled = y_labeled.to(device)
        x_unlabeled = x_unlabeled.to(device)

        optimizer.zero_grad()

        # Doing the forward pass for labeled data
        y_onehot = F.one_hot(y_labeled, model.num_classes).float().to(device)
        recon_labeled, mean_l, log_var_l, logits_l = model(x_labeled, y_onehot=y_onehot)
        loss_labeled = criterion_labeled(recon_labeled, x_labeled, mean_l, log_var_l, logits_l, y_labeled)

        # Doing the forward pass for unlabeled data
        recon_unlabeled, mean_u, log_var_u, logits_u = model(x_unlabeled)
        loss_unlabeled = criterion_unlabeled(recon_unlabeled, x_unlabeled, mean_u, log_var_u, logits_u)

        # Combining the losses
        loss = loss_labeled + loss_unlabeled
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()

    return tot_loss


def main():
    # Parsing command-line arguments
    parser = argparse.ArgumentParser(description="Train the M2 VAE model with optimized or standard loss")
    parser.add_argument("--optimized_elbo", action="store_true", help="Use optimized ELBO loss")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for CE loss")
    parser.add_argument("--optimized_label_weight", type=float, default=0.1, help="Weight for optimizeded labels")
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["CIFAR10", "MNIST"], help="Dataset to use")
    args = parser.parse_args()

    # Hyperparameters
    latent_dim = 50
    num_classes = 10
    batch_size = 128
    num_epochs = 50
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # preparing dataset
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
    os.makedirs(dataset_path, exist_ok=True)

    if args.dataset == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.MNIST(root=dataset_path, train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root=dataset_path, train=False, transform=transform, download=True)
        model = MNISTVariationalAutoEncoder(latent_dim, num_classes).to(device)
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.CIFAR10(root=dataset_path, train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(root=dataset_path, train=False, transform=transform, download=True)
        model = CIFAR10VariationalAutoEncoder(latent_dim, num_classes).to(device)

    labeled_size = int( 0.1*len(dataset) )

    unlabeled_size = len(dataset) - labeled_size

    (labeled_data, unlabeled_data) = random_split(dataset, [labeled_size, unlabeled_size])

    labeled_loader = DataLoader(labeled_data, batch_size=batch_size, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Define criteria using lamda function for labeled  and unlabeled data 
    if args.optimized_elbo:
        criterion_labeled = lambda *inputs: optimized_ELBO_labeled(*inputs, alpha=args.alpha, optimized_label_weight=args.optimized_label_weight)
        criterion_unlabeled = lambda *inputs: optimized_ELBO_unlabeled(*inputs, alpha=args.alpha)
    else:
        criterion_labeled = lambda *inputs: m2_loss_labeled(*inputs, alpha=args.alpha)
        criterion_unlabeled = lambda *inputs: m2_loss_unlabeled(*inputs, alpha=args.alpha)

    # Training and validation loop
    for epoch in range(1, num_epochs + 1):
        tot_loss = train(model, labeled_loader, unlabeled_loader, optimizer, criterion_labeled, criterion_unlabeled, epoch, device)
        acc = validate(model, test_loader, device)
        print(f"Epoch {epoch}: Train Loss: {tot_loss:.4f}: Validation acc: {acc:.2f}%")

    # Save the trained model based on the loss type
    trained_model_dir = os.path.abspath(os.path.join("..", "trained_models"))
    os.makedirs(trained_model_dir, exist_ok=True)

    if args.optimized_elbo:
        save_path = os.path.join(trained_model_dir, "optimized_vae_model_final.pth")
    else:
        save_path = os.path.join(trained_model_dir, "m2_model_final.pth")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")



if __name__ == "__main__":
    main()
