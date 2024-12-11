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
