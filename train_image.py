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



def get_parser():
    parser = argparse.ArgumentParser(description="Image classification pipeline for alpha beta tuning")
    parser.add_argument("--dataset", type=str, required=True, choices=["mnist", "cifar10", "celeba"], help="Dataset to use for training")
    parser.add_argument("--model", type=str, required=True, choices=["vgg19", "resnet50", "vit"], help="Model architecture to use for training")
    parser.add_argument("--alpha", type=float, required=True, help="Alpha value clean/noisy split")
    parser.add_argument("--adaptive", action="store_true", help="Use adaptive batch size for training")
    parser.add_argument("--batch_size", type=int, default=32, help = "Initial batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help = "Number of epochs for training")
    parser.add_argument("--lr", type=float, default=0.001, help = "Learning rate for training")
    parser.add_argument("--seed", type=int, default=42, help = "Random seed for training")
    parser.add_argument("--output_dir", type=str, default="results", help = "Directory to save output")
    return parser

# class CelebADataset(torch.utils.data.Dataset):
#     def __init__(self, img_dir, metadata, transform=None):
#         self.img_dir = img_dir
#         self.metadata = metadata
#         self.transform = transform

#     def __len__(self):
#         return len(self.metadata)
    
#     def __getitem__(self, idx):
#         img_name = self.metadata.iloc[idx, 0]
#         label = self.metadata.iloc[idx, 1:]
#         img_path = os.path.join(self.img_dir, img_name)
#         image = Image.open(img_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         return image, label

def prepare_dataset(dataset_name, alpha):
    transform_clean = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])

    transform_noisy = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1), # we add gaussian noise to the image
        # salt and pepper noise with 10% probability
        transforms.Lambda(lambda x: x * (torch.rand_like(x) > 0.1).float() + (torch.rand_like(x) < 0.1).float()),
        # random occlusion with 10% probability
        transforms.Lambda(lambda x: x.masked_fill(torch.rand_like(x) < 0.1, 0)),
        # Gaussian blur
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2)),
        transforms.Normalize((0.5,),(0.5,))
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

    num_samples = len(dataset_clean)
    noisy_size = int(num_samples * alpha)
    indices = np.random.permutation(num_samples)
    noisy_indices = indices[:noisy_size]
    clean_indices = indices[noisy_size:]

    dataset_clean = Subset(dataset_clean, clean_indices)
    dataset_noisy = Subset(dataset_noisy, noisy_indices)

    return dataset_clean, dataset_noisy, dataset_test

def get_model(model_name, num_classes):
    if model_name == "vgg19":
        model = models.vgg19(pretrained=False)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "vit":
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported")
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, adaptive, batch_size):
    model.to(device)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if adaptive:
                adaptive_batch_size = max(1, int(batch_size * (1.0 - epoch / epochs)))
                train_loader.batch_size = adaptive_batch_size

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

    dataset_clean, dataset_noisy, dataset_test = prepare_dataset(args.dataset, args.alpha)

    train_loader_phase1 = DataLoader(dataset_noisy, batch_size=args.batch_size, shuffle=True)
    train_loader_phase2 = DataLoader(dataset_clean, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    if args.dataset == "mnist":
        num_classes = 10
    elif args.dataset == "cifar10":
        num_classes = 10
    elif args.dataset == "celeba":
        num_classes = 40

    model = get_model(args.model, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("Starting Phase 1 Training with Noisy Data")
    history_phase1 = train_model(model, train_loader_phase1, val_loader, criterion, optimizer, device, args.epochs, args.adaptive, args.batch_size)

    print("Starting Phase 2 Training with Clean Data")
    history_phase2 = train_model(model, train_loader_phase2, val_loader, criterion, optimizer, device, args.epochs, args.adaptive, args.batch_size)

    output_name = f"{args.dataset}_{args.model}_alpha_{args.alpha}_adaptive_{args.adaptive}_batch_size_{args.batch_size}_epochs_{args.epochs}_lr_{args.lr}_seed_{args.seed}"
    output_dir = os.path.join(args.output_dir, output_name)
    save_results(output_dir, output_name, {"phase1": history_phase1, "phase2": history_phase2})

if __name__ == "__main__":
    main()

