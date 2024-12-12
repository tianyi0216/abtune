import argparse
import json
import numpy as np
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import random

def get_parser():
    parser = argparse.ArgumentParser(description="Text classification pipeline for alpha beta tuning")
    parser.add_argument("--model", type=str, default="bert-base-uncased", help="Model architecture")
    parser.add_argument("--alpha", type=float, required=True, help="Alpha value for clean/noisy split")
    parser.add_argument("--adaptive", action="store_true", help="Use adaptive batch size for training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--output_dir", type=str, default="results_text", help="Directory to save output")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, transform=None):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample["data"]["text"]
        label = sample["label"]

        if self.transform:
            text = self.transform(text)

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }
    
# Text data preparation function
def prepare_text_dataset(json_path, alpha, tokenizer, max_len=128, transform_noisy=None):
    with open(json_path, "r") as f:
        dataset = json.load(f)

    # Extract clean and noisy data
    clean_data = [{"data": v["data"], "label": v["label"]} for k, v in dataset.items()]
    noisy_data = [{"data": v["data"], "label": max(set(v["weak_labels"]), key=v["weak_labels"].count)} for k, v in dataset.items()]

    # if max is -1, change to 0
    for item in noisy_data:
        if item["label"] == -1:
            item["label"] = 0

    # Total samples
    total_samples = 1000
    noisy_count = int(total_samples * alpha)
    clean_count = total_samples - noisy_count

    # Randomly sample clean and noisy datasets
    clean_sample = random.sample(clean_data, min(clean_count, len(clean_data)))
    noisy_sample = random.sample(noisy_data, min(noisy_count, len(noisy_data)))

    # Apply transformation to noisy samples, if any
    if transform_noisy:
        for item in noisy_sample:
            item["data"]["text"] = transform_noisy(item["data"]["text"])

    # Test set (remaining clean samples not in train)
    remaining_clean_data = [item for item in clean_data if item not in clean_sample]
    test_sample = random.sample(remaining_clean_data, min(200, len(remaining_clean_data)))

    return (
        TextDataset(clean_sample, tokenizer, max_len),
        TextDataset(noisy_sample, tokenizer, max_len),
        TextDataset(test_sample, tokenizer, max_len),
    )

def noisy_transform(text):
    """Simulate noise in text data."""
    # Randomly add typos
    if random.random() > 0.8:
        text = text.replace("e", "3").replace("o", "0").replace("a", "@")
    # Add random phrases
    if random.random() > 0.9:
        text += " !!!! Cmon"
    return text

def train_model(model, train_loader, val_loader, optimizer, device, epochs, adaptive, batch_size, dataset, seed):
    # seed
    torch.manual_seed(seed)
    model.to(device)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        if adaptive:
            adaptive_batch_size = max(1, int(batch_size * (1.0 - epoch / epochs)))
            train_loader = DataLoader(dataset, batch_size=adaptive_batch_size, shuffle=True, drop_last=True)
        else:
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        model.train()
        running_loss = 0.0
        
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
            if i == 0:
                print("Current batch size: ", len(batch))
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        val_loss, val_acc = evaluate_model(model, val_loader, device)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    return history

# Evaluation Function
def evaluate_model(model, val_loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            running_loss += loss.item()
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy

def save_results(output_dir, output_name, history):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_dir, f"{output_name}.json"), "w") as f:
        json.dump(history, f)

def main():
    parser = get_parser()
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if device != "cuda":
    #     device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Using {device} for training")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    max_len = 512

    dataset_clean, dataset_noisy, dataset_test = prepare_text_dataset("youtube_raw/train.json", args.alpha, tokenizer, max_len, noisy_transform)
    print(f"Dataset loaded")

    print("Length of dataset_clean: ", len(dataset_clean))
    print("Length of dataset_noisy: ", len(dataset_noisy))
    print("Length of dataset_test: ", len(dataset_test))

    if args.alpha == 0.0:
        train_loader = DataLoader(dataset_clean, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=True)

        print("Dataloader created")

        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
        optimizer = AdamW(model.parameters(), lr=args.lr)

        print("Model created, begin training")
        history = train_model(model, train_loader, val_loader, optimizer, device, args.epochs, args.adaptive, args.batch_size, dataset_clean, args.seed)

        print("Training finished")

        output_name = f"Text_alpha_{args.alpha}_adaptive_{args.adaptive}"
        output_dir = os.path.join(args.output_dir, output_name)
        save_results(output_dir, output_name, {"phase1": history})

    elif args.alpha == 1.0:
        train_loader = DataLoader(dataset_noisy, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=True)

        print("Dataloader created")

        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
        optimizer = AdamW(model.parameters(), lr=args.lr)

        print("Model created, begin training")
        history = train_model(model, train_loader, val_loader, optimizer, device, args.epochs, args.adaptive, args.batch_size, dataset_noisy, args.seed)

        print("Training finished")

        output_name = f"Text_alpha_{args.alpha}_adaptive_{args.adaptive}"
        output_dir = os.path.join(args.output_dir, output_name)
        save_results(output_dir, output_name, {"phase1": history})

    else:
        train_loader_phase1 = DataLoader(dataset_noisy, batch_size=args.batch_size, shuffle=True, drop_last=True)
        train_loader_phase2 = DataLoader(dataset_clean, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=True)

        print("Dataloader created")

        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
        optimizer = AdamW(model.parameters(), lr=args.lr)

        print("Starting Phase 1 Training with Noisy Data")
        history_phase1 = train_model(model, train_loader_phase1, val_loader, optimizer, device, int(args.epochs//2), args.adaptive, args.batch_size, dataset_noisy, args.seed)

        print("Starting Phase 2 Training with Clean Data")
        history_phase2 = train_model(model, train_loader_phase2, val_loader, optimizer, device, int(args.epochs//2), args.adaptive, args.batch_size, dataset_clean, args.seed)

        output_name = f"Text_alpha_{args.alpha}_adaptive_{args.adaptive}"
        output_dir = os.path.join(args.output_dir, output_name)
        save_results(output_dir, output_name, {"phase1": history_phase1, "phase2": history_phase2})

if __name__ == "__main__":
    main()