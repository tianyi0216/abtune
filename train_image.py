# import the necessary libraries
import argparse
import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import json
from pathlib import Path
from PIL import Image
import pandas as pd
import timm
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig, ViTForImageClassification


def get_parser():
    parser = argparse.ArgumentParser(description="Image classification pipeline for alpha beta tuning")
    parser.add_argument("--dataset", type=str, required=True, choices=["mnist", "cifar10", "celeba"], help="Dataset to use for training")
    parser.add_argument("--model", type=str, required=True, choices=["lenet", "resnet18", "vit"], help="Model architecture to use for training")
    parser.add_argument("--alpha", type=float, required=True, help="Alpha value clean/noisy split")
    parser.add_argument("--adaptive", action="store_true", help="Use adaptive batch size for training")
    parser.add_argument("--batch_size", type=int, default=32, help = "Initial batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help = "Number of epochs for training")
    parser.add_argument("--lr", type=float, default=0.001, help = "Learning rate for training")
    parser.add_argument("--seed", type=int, default=42, help = "Random seed for training")
    parser.add_argument("--output_dir", type=str, default="results", help = "Directory to save output")
    return parser

class LeNet(nn.Module):
    def __init__(self, input_size = 28,num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        # Second convolutional layer: 6 input channels, 16 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 4x4 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        # Convolutional layers with ReLU activation and max pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 2x2 max pooling
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        # Flatten the feature maps
        x = x.view(-1, 16 * 4 * 4)
        # Fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def prepare_dataset(dataset_name, alpha, model_name = None):
    transform_clean = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    transform_noisy = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1), # we add gaussian noise to the image
        # salt and pepper noise with 10% probability
        transforms.Lambda(lambda x: x * (torch.rand_like(x) > 0.1).float() + (torch.rand_like(x) < 0.1).float()),
        # random occlusion with 10% probability
        transforms.Lambda(lambda x: x.masked_fill(torch.rand_like(x) < 0.1, 0)),
        # Gaussian blur
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2)),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    if dataset_name == "cifar10":
        transform_clean = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        transform_noisy = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1), # we add gaussian noise to the image
            # salt and pepper noise with 10% probability
            transforms.Lambda(lambda x: x * (torch.rand_like(x) > 0.1).float() + (torch.rand_like(x) < 0.1).float()),
            # random occlusion with 10% probability
            transforms.Lambda(lambda x: x.masked_fill(torch.rand_like(x) < 0.1, 0)),
            # Gaussian blur
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2)),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

    # load dataset
    if dataset_name == "mnist":
        dataset_clean = datasets.MNIST(root='data', train=True, download=True, transform=transform_clean)
        dataset_noisy = datasets.MNIST(root='data', train=True, download=True, transform=transform_noisy)
        dataset_test = datasets.MNIST(root='data', train=False, download=True, transform=transform_clean)
    elif dataset_name == "cifar10":
        dataset_clean = datasets.CIFAR10(root='data', train=True, download=True, transform=transform_clean)
        dataset_noisy = datasets.CIFAR10(root='data', train=True, download=True, transform=transform_noisy)
        dataset_test = datasets.CIFAR10(root='data', train=False, download=True, transform=transform_clean)
    elif dataset_name == "celeba":
        dataset_clean = datasets.CelebA(root='data', split='train', download=True, transform=transform_clean)
        dataset_noisy = datasets.CelebA(root='data', split='train', download=True, transform=transform_noisy)
        dataset_test = datasets.CelebA(root='data', split='test', download=True, transform=transform_clean)

    #only use 5000 samples
    all_indices = np.random.permutation(len(dataset_clean))
    selected_indices = all_indices[:5000]
    dataset_clean = Subset(dataset_clean, selected_indices)
    dataset_noisy = Subset(dataset_noisy, selected_indices)

    # use 1000 samples for test
    dataset_test = Subset(dataset_test, np.random.permutation(len(dataset_test))[:1000])

    num_samples = len(dataset_clean)
    noisy_size = int(num_samples * alpha)
    indices = np.random.permutation(num_samples)
    noisy_indices = indices[:noisy_size]
    clean_indices = indices[noisy_size:]

    dataset_clean = Subset(dataset_clean, clean_indices)
    dataset_noisy = Subset(dataset_noisy, noisy_indices)

    return dataset_clean, dataset_noisy, dataset_test

def get_model(model_name, num_classes):
    if model_name == "lenet":
        model = LeNet(num_classes)
    elif model_name == "resnet18":
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "vit":
        class ViT(nn.Module):
            def __init__(self, num_classes):
                super(ViT, self).__init__()
                config = ViTConfig(
                image_size=28,        # Input image size
                patch_size=7,         # Smaller patch size
                num_channels=3,       # MNIST will need conversion to 3 channels
                num_labels=num_classes
                )
                self.vit = ViTForImageClassification(config)
                self.vit.classifier = nn.Linear(config.hidden_size, num_classes)
            def forward(self, x):
                outputs = self.vit(x)
                return outputs.logits
        model = ViT(num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported")
    return model

def train_model(model, dataset, train_loader, val_loader, criterion, optimizer, device, epochs, adaptive, batch_size):
    model.to(device)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in tqdm(range(epochs), desc=f"Currently training"):
        if adaptive:
            adaptive_batch_size = max(1, int(batch_size * (1.0 - epoch / epochs)))
            train_loader = DataLoader(dataset, batch_size=adaptive_batch_size, shuffle=True, drop_last=True)
        else:
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if batch_idx == 0:
                print("Current batch size: ", len(inputs))
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    return history

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy

def save_results(output_dir, output_name, history):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results_path = os.path.join(output_dir, f"{output_name}.json")
    with open(results_path, "w") as f:
        json.dump(history, f)

def main():
    parser = get_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device != "cuda":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Using {device} for training")

    dataset_clean, dataset_noisy, dataset_test = prepare_dataset(args.dataset, args.alpha, args.model)
    print(f"Dataset loaded: {args.dataset}")
    if args.dataset == "mnist":
        num_classes = 10
    elif args.dataset == "cifar10":
        num_classes = 10
    elif args.dataset == "celeba":
        num_classes = 40

    if args.alpha == 0.0:
        # only train 1 phase in this case
        train_loader = DataLoader(dataset_clean, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=True)

        print("Dataloader created")

        model = get_model(args.model, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        print("Model created, begin training")
        history = train_model(model, dataset_clean, train_loader, val_loader, criterion, optimizer, device, args.epochs, args.adaptive, args.batch_size)

        print("Training finished")

        output_name = f"{args.dataset}_{args.model}_alpha_{args.alpha}_adaptive_{args.adaptive}_batch_size_{args.batch_size}_epochs_{args.epochs}_lr_{args.lr}_seed_{args.seed}"
        output_dir = os.path.join(args.output_dir, output_name)
        save_results(output_dir, output_name, {"phase1": history})

    elif args.alpha == 1.0:
        # only train 1 phase in this case
        train_loader = DataLoader(dataset_noisy, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=True)

        print("Dataloader created")

        model = get_model(args.model, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        print("Model created, begin training")
        history = train_model(model, dataset_noisy, train_loader, val_loader, criterion, optimizer, device, args.epochs, args.adaptive, args.batch_size)
        print("Training finished")

        output_name = f"{args.dataset}_{args.model}_alpha_{args.alpha}_adaptive_{args.adaptive}_batch_size_{args.batch_size}_epochs_{args.epochs}_lr_{args.lr}_seed_{args.seed}"
        output_dir = os.path.join(args.output_dir, output_name)
        save_results(output_dir, output_name, {"phase1": history})

    else:
        train_loader_phase1 = DataLoader(dataset_noisy, batch_size=args.batch_size, shuffle=True, drop_last=True)
        train_loader_phase2 = DataLoader(dataset_clean, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=True)

        print("Dataloader created")

        model = get_model(args.model, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        print("Starting Phase 1 Training with Noisy Data")
        history_phase1 = train_model(model, dataset_noisy, train_loader_phase1, val_loader, criterion, optimizer, device, int(args.epochs//2), args.adaptive, args.batch_size)

        print("Starting Phase 2 Training with Clean Data")
        history_phase2 = train_model(model, dataset_clean, train_loader_phase2, val_loader, criterion, optimizer, device, int(args.epochs//2), args.adaptive, args.batch_size)

        output_name = f"{args.dataset}_{args.model}_alpha_{args.alpha}_adaptive_{args.adaptive}_batch_size_{args.batch_size}_epochs_{args.epochs}_lr_{args.lr}_seed_{args.seed}"
        output_dir = os.path.join(args.output_dir, output_name)
        save_results(output_dir, output_name, {"phase1": history_phase1, "phase2": history_phase2})

if __name__ == "__main__":
    main()

