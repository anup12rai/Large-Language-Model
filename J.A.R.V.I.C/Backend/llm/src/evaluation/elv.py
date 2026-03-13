import torch
from torch.utils.data import DataLoader, TensorDataset
from src.preprocessing.datasets import get_train_val_split
from src.preprocessing.tokenizer import load_tokenizer
from src.preprocessing.encode_data import encode_texts
from src.model.transformer import SimpleTransformer
from src.model.classifier import QueryClassifier
from src.model.model_utils import load_model
from src.evaluation.metrics import compute_metrics
from src.training.config import config

DEVICE = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = config.get("max_seq_len", 32)

def evaluate(model_path, tokenizer_path):
    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_path)
    _, val_texts, _, val_labels = get_train_val_split(test_size=config.get("train_val_split_ratio", 0.1))
    
    X_val = encode_texts(tokenizer, val_texts, max_len=MAX_LEN)
    y_val = torch.tensor(val_labels)
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=config.get("batch_size", 16))
    transformer = SimpleTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        embed_dim=config.get("embed_dim", 64),
        num_heads=config.get("num_heads", 4),
        ff_hidden=config.get("ff_hidden", 128),
        num_layers=config.get("num_layers", 2),
        max_len=MAX_LEN
    )
    model = QueryClassifier(transformer)
    model = load_model(model, model_path, device=DEVICE)
    model.to(DEVICE)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            logits = model(X_batch)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    metrics = compute_metrics(all_labels, all_preds)
    print("Evaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-score: {metrics['f1_score']:.4f}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
    return metrics

# Example usage
if __name__ == "__main__":
    evaluate(model_path="../model.pt", tokenizer_path="../data/processed/tokenizer.json")