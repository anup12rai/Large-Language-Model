import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.preprocessing.datasets import get_train_val_split
from src.preprocessing.tokenizer import load_tokenizer
from src.preprocessing.encode_data import encode_texts
from src.model.transformer import SimpleTransformer
from src.model.classifier import QueryClassifier
from src.model.model_utils import save_model
from src.utils.constants import LABELS

# -------------------------------
# Load config.yaml
# -------------------------------
with open("src/training/config.yaml") as f:
    config = yaml.safe_load(f)

BATCH_SIZE = config["batch_size"]
EPOCHS = config["epochs"]
LR = config["learning_rate"]
MAX_LEN = config["max_seq_len"]
DEVICE = config["device"]
MODEL_PATH = config["model_save_path"]

EMBED_DIM = config["embed_dim"]
NUM_HEADS = config["num_heads"]
FF_HIDDEN = config["ff_hidden"]
NUM_LAYERS = config["num_layers"]

TOKENIZER_PATH = config["tokenizer_path"]

# -------------------------------
# Prepare dataloaders
# -------------------------------
def prepare_dataloader():
    train_texts, val_texts, train_labels, val_labels = get_train_val_split(test_size=config["train_val_split_ratio"])
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    
    X_train = encode_texts(tokenizer, train_texts, max_len=MAX_LEN)
    X_val = encode_texts(tokenizer, val_texts, max_len=MAX_LEN)
    y_train = torch.tensor(train_labels)
    y_val = torch.tensor(val_labels)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    return train_loader, val_loader, tokenizer

# -------------------------------
# Training function
# -------------------------------
def train():
    train_loader, val_loader, tokenizer = prepare_dataloader()

    # Initialize model
    transformer = SimpleTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_hidden=FF_HIDDEN,
        num_layers=NUM_LAYERS,
        max_len=MAX_LEN
    )
    model = QueryClassifier(transformer)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        val_acc = correct / total

        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f} - Val Accuracy: {val_acc:.4f}")

    # Save model
    save_model(model, MODEL_PATH)
    print(f"Model saved at {MODEL_PATH}")

# -------------------------------
# Run training
# -------------------------------
if __name__ == "__main__":
    train()